import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from utils.utils import get_metrics
from utils.FreDF import FreDFLoss, FreDFLossWithEncouraging, MSEWithEncouragingLoss, GradientNormalizationLoss, FreqGradientLoss

import inspect



# class TemporalBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding):
#         super(TemporalBlock, self).__init__()
#         self.padding = 0
#
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
#                                stride=stride, padding=self.padding, dilation=dilation)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
#                                stride=stride, padding=self.padding, dilation=dilation)
#         self.net = nn.Sequential(self.conv1, self.relu, self.conv2, self.relu)
#         self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
#         self.init_weights()
#
#     def init_weights(self):
#         self.conv1.weight.data.normal_(0, 0.01)
#         self.conv2.weight.data.normal_(0, 0.01)
#         if self.downsample is not None:
#             self.downsample.weight.data.normal_(0, 0.01)
#
#     def forward(self, x):
#         out = self.net(x)
#         res = x if self.downsample is None else self.downsample(x)
#
#         # Adjusting the residual to match the output's size
#         if out.size(2) != res.size(2):
#             res = res[:, :, :out.size(2)]
#         return self.relu(out + res)
#
# class TCN(nn.Module):
#     def __init__(self, config):
#         super(TCN, self).__init__()
#         num_inputs = config.num_chan_eeg  # Number of input channels
#         num_channels = [16, 16]  # Number of channels for each layer
#         kernel_size = 2  # Kernel size for the convolutions
#         output_size = config.num_chan_kin  # Desired output size
#         layers = []
#         num_levels = len(num_channels)
#         for i in range(num_levels):
#             dilation_size = 2 ** i
#             in_channels = num_inputs if i == 0 else num_channels[i-1]
#             out_channels = num_channels[i]
#             layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
#                                      padding=(kernel_size-1) * dilation_size)]
#
#         self.network = nn.Sequential(*layers)
#         self.linear = nn.Linear(num_channels[-1], output_size)
#
#     def forward(self, x):
#         x = x.transpose(1, 2)
#         x = self.network(x)
#         # Taking the last time step's features for each sample in the batch
#         x = x[:, :, -1]
#         return self.linear(x)

# ===== TCN related =====
# Building Temporal Convolutional Networks (TCN)
# Code from https://github.com/locuslab/TCN






class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # X: (Batch, C, T)
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Defining a TCN based on the above class


class TCN(nn.Module):
    def __init__(self, args):
        super(TCN, self).__init__()
        num_channels = [args.tcn.num_hidden] * args.tcn.num_layers
        self.tcn = TemporalConvNet(num_inputs=args.num_chan_eeg,
                                   num_channels=num_channels,
                                   kernel_size=args.tcn.kernel_size,
                                   dropout=args.tcn.dropout
                                   )
        self.fc = nn.Linear(num_channels[-1], args.num_chan_kin)
        if args.eegnet.loss_name == 'mse':
            self.loss_fn = nn.MSELoss()
        elif args.eegnet.loss_name == 'freq_mse':
            self.loss_fn = FreDFLoss()
        elif args.eegnet.loss_name == 'reward_mse':
            self.loss_fn = MSEWithEncouragingLoss()
        elif args.eegnet.loss_name == 'freq_reward_mse':
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
        inputs = batch['EEG']
        Y = batch['Y']
        s = self.tcn(inputs.transpose(1, 2))
        x = self.fc(s[:, :, -1])
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

        return s
