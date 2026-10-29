"""
Based on https://github.com/deepBrains/TSception/blob/master/Models.py

The original model was used for the classification task, and we made changes to apply it to the regression task.

For more details about the models, please refer to original paper:

Yi Ding, Neethu Robinson, Qiuhao Zeng, Dou Chen, Aung Aung Phyo Wai, Tih-Shih Lee, Cuntai Guan,
"TSception: A Deep Learning Framework for Emotion Detection Useing EEG"(IJCNN 2020)

"""
import torch
import torch.nn as nn
import os
import torch
import torch.nn as nn
from utils.myaml import load_config
from utils.FreDF import FreDFLoss, FreDFLossWithEncouraging, MSEWithEncouragingLoss, GradientNormalizationLoss, FreqGradientLoss

import inspect
from utils.utils import get_metrics
from utils.FreDF import FreDFLoss, FreDFLossWithEncouraging, MSEWithEncouragingLoss, GradientNormalizationLoss, FreqGradientLoss


class TSception(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step, padding=0),
            nn.LeakyReLU())

    def __init__(self, config):
        # input_size: EEG channel x datapoint
        super(TSception, self).__init__()
        num_joints = config.eegnet.num_chan_kin 
        sampling_rate = config.sampling_rate
        
        num_T = config.tsception.num_T
        num_S = config.tsception.num_S
        input_size = [config.num_chan_eeg, config.eegnet.eeg.time_step]

        self.inception_window = [0.5, 0.25, 0.15]
        self.TK = [int(self.inception_window[idx] * sampling_rate) for idx in range(3)]

        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.conv_block(1, num_T, (1, self.TK[0]))
        self.Tception2 = self.conv_block(1, num_T, (1, self.TK[1]))
        self.Tception3 = self.conv_block(1, num_T, (1, self.TK[2]))

        self.Sception1 = self.conv_block(num_T, num_S, (int(input_size[-2]), 1))
        self.Sception2 = self.conv_block(num_T, num_S, (int(input_size[-2] * 0.5), 1), (int(input_size[-2] * 0.5), 1))
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)

        # torch.__version__ >= 1.8.0
        self.fc = nn.Linear(63900, num_joints)
        if config.eegnet.loss_name == 'mse':
            self.loss_fn = nn.MSELoss()
        elif config.eegnet.loss_name == 'freq_mse':
            self.loss_fn = FreDFLoss()
        elif config.eegnet.loss_name == 'reward_mse':
            self.loss_fn = MSEWithEncouragingLoss()
        elif config.eegnet.loss_name == 'freq_reward_mse':
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
        x = torch.permute(x, [0, 1, 3, 2])
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        out = out.view(out.size()[0], -1)
        x = self.fc(out)

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
    config = load_config('../config.yaml')
    model = TSception(config)
    data = torch.ones((13, 1, 60, 180))

    a = model(data)
    print(a)
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)


