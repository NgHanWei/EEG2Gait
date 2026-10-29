import torch
import torch.nn as nn
from utils.utils import get_metrics
from utils.FreDF import FreDFLoss, FreDFLossWithEncouraging, MSEWithEncouragingLoss, GradientNormalizationLoss, FreqGradientLoss

import inspect

class ResBlock(nn.Module):
    """Convolutional Residual Block 2D
    This block stacks two convolutional layers with batch normalization,
    max pooling, dropout, and residual connection.
    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        stride: stride of the convolutional layers.
        downsample: whether to use a downsampling residual connection.
        pooling: whether to use max pooling.
    Example:
        >>> import torch
        >>> from pyhealth.models import ResBlock2D
        >>>
        >>> model = ResBlock2D(6, 16, 1, True, True)
        >>> input_ = torch.randn((16, 6, 28, 150))  # (batch, channel, height, width)
        >>> output = model(input_)
        >>> output.shape
        torch.Size([16, 16, 14, 75])
    """

    def __init__(
        self, in_channels, out_channels, stride=1, downsample=False, pooling=False
    ):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(3, stride=stride, padding=1)
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)

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


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampleOrNot:
            residual = self.downsample(x)
        out += residual
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out


class FFCL(nn.Module):
    """The whole model is CNN + LSTM. We combine the embeddings and add an FC layer."""

    def __init__(
        self,
        in_channels=16,
        n_classes=6,
        fft=200,
        steps=20,
        sample_length=2000,
        shrink_steps=20,
        loss_name='mse',
    ):
        super(FFCL, self).__init__()
        self.fft = fft
        self.steps = steps
        self.conv1 = ResBlock(in_channels, 32, 2, True, True)
        self.conv2 = ResBlock(32, 64, 2, True, True)
        self.conv3 = ResBlock(64, 128, 2, True, True)
        self.conv4 = ResBlock(128, 256, 2, True, True)

        self.lstm = nn.LSTM(
            input_size=sample_length // shrink_steps,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
        )
        self.shrink_steps = shrink_steps

        self.classifier = nn.Sequential(
            nn.ELU(),
            nn.Linear(256 * 2, n_classes),
        )
        self.loss_name = loss_name
        if self.loss_name == 'mse':
            self.loss_fn = nn.MSELoss()
        elif self.loss_name == 'freq_mse':
            self.loss_fn = FreDFLoss()
        elif self.loss_name == 'reward_mse':
            self.loss_fn = MSEWithEncouragingLoss()
        elif self.loss_name == 'freq_reward_mse':
            self.loss_fn = FreDFLossWithEncouraging()

    def shorten(self, x):
        """
        x: (batch_size, n_channels, length)
        out: (batch_size, n_channels * shrink_steps, length // shrink_steps)
        """
        segments = [
            x[:, :, i :: self.shrink_steps] for i in range(0, self.shrink_steps)
        ]
        return torch.cat(segments, dim=1)

    def torch_stft(self, x):
        signal = []
        for s in range(x.shape[1]):
            spectral = torch.stft(
                x[:, s, :],
                n_fft=self.fft,
                hop_length=self.fft // self.steps,
                win_length=self.fft,
                normalized=True,
                center=True,
                onesided=True,
                return_complex=True,
            )
            signal.append(spectral)
        stacked = torch.stack(signal).permute(1, 0, 2, 3)
        return torch.abs(stacked)

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
        
        e1 = self.torch_stft(x)
        e1 = self.conv1(e1)
        e1 = self.conv2(e1)
        e1 = self.conv3(e1)
        e1 = self.conv4(e1).squeeze(-1).squeeze(-1)

        e2 = self.shorten(x)
        e2 = self.lstm(e2)[0][:, -1]

        e = torch.cat([e1, e2], dim=1)
        x = self.classifier(e)

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
    x = torch.randn(2, 16, 2000)
    model = FFCL(
        in_channels=16,
        n_classes=6,
        fft=200,
        steps=20,
        sample_length=2000,
        shrink_steps=20,
    )
    out = model(x)
    print(out.shape)
