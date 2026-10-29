
"""
Based on https://github.com/HANYIIK/EEGNet-pytorch/blob/master/model.py

Fixed parameter to variable parameter
"""
import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, config):
        super(EEGNet, self).__init__()
        self.drop_out = config.eegnet.dropout
        self.F1 = config.eegnet.f1
        self.F2 = config.eegnet.f2
        self.D = config.eegnet.d
        self.C = config.num_chan_eeg
        self.sr = config.sampling_rate

        self.num_joints = config.num_chan_kin

        temporal_ks = self.sr//2
        
        self.block_1_temporal = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((temporal_ks//2-1, temporal_ks//2, 0, 0)),
            nn.Conv2d(
                in_channels=1,          # input shape (1, C, T)
                out_channels=self.F1,         # num_filters, F1
                kernel_size=(1, temporal_ks),    # filter size, 0.5 * sampling_rate
                bias=False
            ),                          # output shape (F1, C, T)
            nn.BatchNorm2d(self.F1)           # output shape (F1, C, T)
        )
        
        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_1_depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels=self.F1,          # input shape (F1, C, T)
                out_channels=self.D*self.F1,        # num_filters
                kernel_size=(self.C, 1),    # filter size
                groups=self.F1,
                bias=False
            ),                          # output shape (D*F1, 1, T)
            nn.BatchNorm2d(self.D*self.F1),         # output shape (D*F1, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, self.sr//32)),       # output shape (D*F1, 1, T//?) 
                                                  # reduce the sampling rate of the signal to 32Hz
            nn.Dropout(self.drop_out)   # output shape (D*F1, 1, T//?) ? = 3 if sr=100hz
        )
        
        self.block_2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
               in_channels=self.D*self.F1,          # input shape (D*F1, 1, T//?)
               out_channels=self.F2,         # num_filters
               kernel_size=(1, 16),     # filter size
               groups=self.D*self.F1,
               bias=False
            ),                          # output shape (F2, 1, T//?)
            nn.Conv2d(
                in_channels=self.F2,         # input shape (F2, 1, T//?)
                out_channels=self.F2,        # num_filters
                kernel_size=(1, 1),     # filter size
                bias=False
            ),                          # output shape (F2, 1, T//?)
            nn.BatchNorm2d(self.F2),         # output shape (F2, 1, T//?)
            nn.ELU(),
            nn.AvgPool2d((1, 2)),       # output shape (F2, 1, T//(?*2))
            nn.Dropout(self.drop_out)
        )
        
        self.fc = nn.LazyLinear(self.num_joints)
    
    def forward(self, x):
        x = torch.permute(x, [0,1,3,2])
        x = self.block_1_temporal(x)
        x = self.block_1_depthwise(x)
        x = self.block_2(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  
