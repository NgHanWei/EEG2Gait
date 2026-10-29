from collections import OrderedDict
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_metrics
from utils.FreDF import FreDFLoss, FreDFLossWithEncouraging, MSEWithEncouragingLoss, GradientNormalizationLoss, FreqGradientLoss

import inspect

class DenseLayer(nn.Sequential):
    """Densely connected layer
    Args:
        input_channels: number of input channels
        growth_rate: rate of growth of channels in this layer
        bn_size: multiplicative factor for the bottleneck layer (does not affect the output size)
        drop_rate: dropout rate
        conv_bias: whether to use bias in convolutional layers
        batch_norm: whether to use batch normalization
    Example:
        >>> x = torch.randn(128, 5, 1000)
        >>> batch, channels, length = x.shape
        >>> model = DenseLayer(channels, 5, 2)
        >>> y = model(x)
        >>> y.shape
        torch.Size([128, 10, 1000])
    """

    def __init__(
        self,
        input_channels,
        growth_rate,
        bn_size,
        drop_rate=0.5,
        conv_bias=True,
        batch_norm=True,
    ):
        super(DenseLayer, self).__init__()
        if batch_norm:
            self.add_module("norm1", nn.BatchNorm1d(input_channels)),
        self.add_module("elu1", nn.ELU()),
        self.add_module(
            "conv1",
            nn.Conv1d(
                input_channels,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=conv_bias,
            ),
        ),
        if batch_norm:
            self.add_module("norm2", nn.BatchNorm1d(bn_size * growth_rate)),
        self.add_module("elu2", nn.ELU()),
        self.add_module(
            "conv2",
            nn.Conv1d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=conv_bias,
            ),
        ),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Sequential):
    """Densely connected block
    Args:
        num_layers: number of layers in this block
        input_channls: number of input channels
        growth_rate: rate of growth of channels in this layer
        bn_size: multiplicative factor for the bottleneck layer (does not affect the output size)
        drop_rate: dropout rate
        conv_bias: whether to use bias in convolutional layers
        batch_norm: whether to use batch normalization
    Example:
        >>> x = torch.randn(128, 5, 1000)
        >>> batch, channels, length = x.shape
        >>> model = DenseBlock(3, channels, 5, 2)
        >>> y = model(x)
        >>> y.shape
        torch.Size([128, 20, 1000])
    """

    def __init__(
        self,
        num_layers,
        input_channels,
        growth_rate,
        bn_size,
        drop_rate=0.5,
        conv_bias=True,
        batch_norm=True,
    ):
        super(DenseBlock, self).__init__()
        for idx_layer in range(num_layers):
            layer = DenseLayer(
                input_channels + idx_layer * growth_rate,
                growth_rate,
                bn_size,
                drop_rate,
                conv_bias,
                batch_norm,
            )
            self.add_module("denselayer%d" % (idx_layer + 1), layer)


class TransitionLayer(nn.Sequential):
    """pooling transition layer
    Args:
        input_channls: number of input channels
        output_channels: number of output channels
        conv_bias: whether to use bias in convolutional layers
        batch_norm: whether to use batch normalization
    Example:
        >>> x = torch.randn(128, 5, 1000)
        >>> model = TransitionLayer(5, 18)
        >>> y = model(x)
        >>> y.shape
        torch.Size([128, 18, 500])
    """

    def __init__(
        self, input_channels, output_channels, conv_bias=True, batch_norm=True
    ):
        super(TransitionLayer, self).__init__()
        if batch_norm:
            self.add_module("norm", nn.BatchNorm1d(input_channels))
        self.add_module("elu", nn.ELU())
        self.add_module(
            "conv",
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=1,
                stride=1,
                bias=conv_bias,
            ),
        )
        self.add_module("pool", nn.AvgPool1d(kernel_size=2, stride=2))


class SPaRCNet(nn.Module):
    """
    1D CNN model for biosignal classification

    Jing, Jin, Wendong Ge, Shenda Hong, Marta Bento Fernandes, Zhen Lin, Chaoqi Yang, Sungtae An et al. "Development of expert-level classification of seizures
        and rhythmic and periodic patterns during EEG interpretation." Neurology 100, no. 17 (2023): e1750-e1762.

    @article{jing2023development,
    title={Development of expert-level classification of seizures and rhythmic and periodic patterns during EEG interpretation},
    author={Jing, Jin and Ge, Wendong and Hong, Shenda and Fernandes, Marta Bento and Lin, Zhen and Yang, Chaoqi and An, Sungtae and Struck, Aaron F and Herlopian, Aline and Karakis, Ioannis and others},
    journal={Neurology},
    volume={100},
    number={17},
    pages={e1750--e1762},
    year={2023},
    publisher={AAN Enterprises}
    }

    """

    def __init__(
        self,
        in_channels: int = 16,
        sample_length: int = 2000,
        n_classes: int = 2,
        block_layers=4,
        growth_rate=16,
        bn_size=16,
        drop_rate=0.5,
        conv_bias=True,
        batch_norm=True,
        loss_name='mse',
        **kwargs,
    ):
        super(SPaRCNet, self).__init__()

        # add initial convolutional layer
        out_channels = 2 ** (math.floor(np.log2(in_channels)) + 1)
        first_conv = OrderedDict(
            [
                (
                    "conv0",
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=7,
                        stride=2,
                        padding=3,
                        bias=conv_bias,
                    ),
                )
            ]
        )
        first_conv["norm0"] = nn.BatchNorm1d(out_channels)
        first_conv["elu0"] = nn.ELU()
        first_conv["pool0"] = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.encoder = nn.Sequential(first_conv)

        n_channels = out_channels

        # add dense blocks
        for n_layer in np.arange(math.floor(np.log2(sample_length // 4))):
            block = DenseBlock(
                num_layers=block_layers,
                input_channels=n_channels,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                conv_bias=conv_bias,
                batch_norm=batch_norm,
            )
            self.encoder.add_module("denseblock%d" % (n_layer + 1), block)
            # update number of channels after each dense block
            n_channels = n_channels + block_layers * growth_rate

            trans = TransitionLayer(
                input_channels=n_channels,
                output_channels=n_channels // 2,
                conv_bias=conv_bias,
                batch_norm=batch_norm,
            )
            self.encoder.add_module("transition%d" % (n_layer + 1), trans)
            # update number of channels after each transition layer
            n_channels = n_channels // 2

        """ classification layer """
        self.classifier = nn.Sequential(
            nn.ELU(),
            nn.Linear(n_channels, n_classes),
        )

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.loss_name = loss_name
        if self.loss_name == 'mse':
            self.loss_fn = nn.MSELoss()
        elif self.loss_name == 'freq_mse':
            self.loss_fn = FreDFLoss()
        elif self.loss_name == 'reward_mse':
            self.loss_fn = MSEWithEncouragingLoss()
        elif self.loss_name == 'freq_reward_mse':
            self.loss_fn = FreDFLossWithEncouraging()
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


    def forward(self, batch, metrics=None):
        # x = self.allButLastLayers(x)
        x = batch['EEG']
        Y = batch['Y']
        emb = self.encoder(x).squeeze(-1)
        x = self.classifier(emb)
        loss = self.loss_fn(x, Y)
        if not self.training:
            return loss.item(), x.to(torch.float32)
        with torch.amp.autocast('cuda', enabled=False):
            if 'r2' in metrics:
                results = get_metrics(x.to(torch.float32).detach().cpu().numpy(), Y.cpu().numpy(), metrics, is_binary=False)
        log = {}
        split="train" if self.training else "val"
        log[f'{split}/total_loss'] = loss.item()
        for key, value in results.items():
            log[f'{split}/{key}'] = value

        return loss, log


if __name__ == "__main__":
    X = torch.randn(2, 16, 2000)
    model = SPaRCNet(in_channels=16, sample_length=2000)
    out = model(X)
    print(out.shape)
