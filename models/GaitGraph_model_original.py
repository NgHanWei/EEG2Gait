import torch
import torch.nn as nn

from utils.myaml import load_config
import torch.nn.functional as F
import time
import math
from models.tcn_model import TemporalBlock, TemporalConvNet
import numpy as np
from .DCN_model import Conv2dWithConstraint
from einops import rearrange
from .EmT import GraphEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.FreDF import FreDFLoss, FreDFLossWithEncouraging, MSEWithEncouragingLoss, GradientNormalizationLoss, FreqGradientLoss

from utils.utils import get_metrics
import inspect

class GaitGraph_tmp(nn.Module):
    """
    Modified GaitGraph with GraphNet block (GCN).
    """

    def __init__(self, config, init_adj=False):
        super(GaitGraph_tmp, self).__init__()
        self.config = config.eegnet
        nChan = self.config.num_chan_eeg
        print('nChan:', nChan)
        nTime = self.config.eeg.time_step
        pool_width = 3
        poolSize = {
            "LocalLayers": [(1, pool_width), (1, pool_width), (1, pool_width)],
            "GlobalLayers": (1, pool_width),
        }
        kernel_width = self.config.eeg.time_step // 20
        localKernalSize = {
            "LocalLayers": [(1, kernel_width), (1, kernel_width), (1, kernel_width)],
            "GlobalLayers": (1, kernel_width),
        }
        nClass = config.eegnet.num_chan_kin
        dropoutP = 0.5
        nFilt_FirstLayer = 25
        self.nFilt_FirstLayer = nFilt_FirstLayer
        nFiltLaterLayer = [25, 50, 100, 200]

        self.firstLayer = nn.Sequential(

            nn.ZeroPad2d((localKernalSize["LocalLayers"][0][1] // 2 - 1, localKernalSize["LocalLayers"][0][1] // 2, 0, 0)),
            Conv2dWithConstraint(
                1, nFilt_FirstLayer, localKernalSize["LocalLayers"][0], padding=0, max_norm=2),

        )

        graph2token = 'Flatten'
        encoder_type = 'GCN'
        layers_graph = [1, 2]
        K = 4
        if init_adj:

            repeated_adj_matrix = self.config.adj_matrix.unsqueeze(0).repeat(len(layers_graph), 1, 1)
            self.adjs = nn.Parameter(repeated_adj_matrix, requires_grad=True)
            print('Initializing adjacency matrix with provided values.')
        else:
            # self.adjs = nn.Parameter(torch.FloatTensor(len(layers_graph), self.config.num_chan_eeg, self.config.num_chan_eeg), requires_grad=True)
            # Initialize adjacency matrix as a random orthogonal matrix for each layer
            adjs = []
            for _ in range(len(layers_graph)):
                # Generate a random matrix and compute its QR decomposition
                rand_mat = torch.randn(self.config.num_chan_eeg, self.config.num_chan_eeg)
                q, _ = torch.linalg.qr(rand_mat)
                adjs.append(q)
            adjs = torch.stack(adjs, dim=0)
            self.adjs = nn.Parameter(adjs, requires_grad=True)
            print('Initializing adjacency matrix with random values.')
        nn.init.xavier_uniform_(self.adjs)

        self.GE1 = GraphEncoder(
            num_layers=layers_graph[0], num_node=self.config.num_chan_eeg, in_features=nFilt_FirstLayer,
            out_features=nFilt_FirstLayer, K=K, graph2token=graph2token, encoder_type=encoder_type
        )
        self.GE2 = GraphEncoder(
            num_layers=layers_graph[1], num_node=self.config.num_chan_eeg, in_features=nFilt_FirstLayer,
            out_features=nFilt_FirstLayer, K=K, graph2token=graph2token, encoder_type=encoder_type
        )
        self.spatialLayer = nn.Sequential(
            Conv2dWithConstraint(25, 25, (nChan, 1), padding=0, bias=False, max_norm=2),
            nn.BatchNorm2d(nFilt_FirstLayer),
            nn.ELU(),
            nn.MaxPool2d(poolSize["LocalLayers"][0], stride=poolSize["LocalLayers"][0]),
        )
        self.middleLayers = nn.Sequential(
            *[
                self.convBlock(inF, outF, dropoutP, kernalS, poolS)
                for inF, outF, kernalS, poolS in zip(
                    nFiltLaterLayer[:-1],
                    nFiltLaterLayer[1:-1],
                    localKernalSize["LocalLayers"][1:],
                    poolSize["LocalLayers"][1:],
                )
            ]
        )
        
        self.tmpdropout = nn.Dropout(p=dropoutP)
        self.firstGlobalLayer = self.convBlock(
            nFiltLaterLayer[-2],
            nFiltLaterLayer[-1],
            dropoutP,
            localKernalSize["GlobalLayers"],
            poolSize["GlobalLayers"],
        )

        # self.allButLastLayers = nn.Sequential(
        #     self.firstLayer, self.middleLayers, self.firstGlobalLayer
        # )

        # self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], nClass, (1, nTime // 81))


        # self.weight_keys = [
        #     [
        #         "allButLastLayers.0.0.weight",
        #         "allButLastLayers.0.0.bias",
        #         "allButLastLayers.0.1.weight",
        #     ],
        #     ["allButLastLayers.1.0.1.weight"],
        #     ["allButLastLayers.1.1.1.weight"],
        #     ["allButLastLayers.2.1.weight"],
        #     ["lastLayer.0.weight", "lastLayer.0.bias"],
        # ]
        self.temporalGlobalLayer = nn.MultiheadAttention(embed_dim=200, num_heads=4)

        self.temporalLastLayer = Conv2dWithConstraint(nFiltLaterLayer[-1], nClass, (1, nTime // 81 * 2), padding=0, max_norm=2)
        if self.config.loss_name == 'mse':
            self.loss_fn = nn.MSELoss()
        elif self.config.loss_name == 'freq_mse':
            self.loss_fn = FreDFLoss()
        elif self.config.loss_name == 'reward_mse':
            self.loss_fn = MSEWithEncouragingLoss()
        elif self.config.loss_name == 'freq_reward_mse':
            self.loss_fn = FreDFLossWithEncouraging()
        else:
            self.loss_fn = nn.MSELoss()

    def get_adj(self, self_loop=True):
        # self.adjs : n, node, node
        num_nodes = self.adjs.shape[-1]
        adj = F.relu(self.adjs + self.adjs.transpose(2, 1))
        if self_loop:
            adj = adj + torch.eye(num_nodes, device=adj.device)
        return adj

    def convBlock(self, inF, outF, dropoutP, kernalSize, poolSize, *args, **kwargs):
        return nn.Sequential(
            nn.Dropout(p=dropoutP),
            nn.ZeroPad2d((kernalSize[1] // 2 - 1, kernalSize[1] // 2, 0, 0)),
            Conv2dWithConstraint(
                inF, outF, kernalSize, bias=False, max_norm=2, *args, **kwargs
            ),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d(poolSize, stride=poolSize),
        )


    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(

            # nn.ZeroPad2d((kernalSize[1] // 2 - 1, kernalSize[1] // 2, 0, 0)),
            Conv2dWithConstraint(inF, outF, kernalSize, max_norm=0.5, *args, **kwargs),
            # nn.LogSoftmax(dim=1),
        )

    def calculateOutSize(self, model, nChan, nTime):
        """
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        """
        data = torch.rand(1, 1, nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def forward(self, batch, metrics=None):
        # x = self.allButLastLayers(x)
        x = batch['EEG'] #(B, 1, 59, 400)
        Y = batch['Y']
        x = self.firstLayer(x) # (B, 25, C, T)
        x_gcn = x.permute(0, 3, 2, 1)  # (B, T, C, 8)
        B, T, C, _ = x_gcn.shape
        x_reshaped = rearrange(x_gcn, 'b t c f -> (b t) c f')
        adjs = self.get_adj()
        # x_reshaped = x_gcn.reshape(B * T, C, self.config.F1)
        x_gcn1 = self.GE1(x_reshaped, adjs[0])
        x_gcn1 = x_gcn1.reshape(B, T * C * self.nFilt_FirstLayer)
        x_gcn1 = x_gcn1.reshape(B, T, C, self.nFilt_FirstLayer)
        x_gcn1 = x_gcn1.permute(0, 3, 2, 1)  # (B, F1, C, T)

        x_gcn2 = self.GE2(x_reshaped, adjs[1])  # (B*T, C*F1)
        x_gcn2 = x_gcn2.reshape(B, T * C * self.nFilt_FirstLayer)
        x_gcn2 = x_gcn2.reshape(B, T, C, self.nFilt_FirstLayer)
        x_gcn2 = x_gcn2.permute(0, 3, 2, 1)  # (B, F1/2, C, T)

        x = x + (x_gcn1 + x_gcn2) / 2 # (B, 25, C, T)
        x = self.spatialLayer(x)# (B, 25, 1, T // 3)

        x = self.middleLayers(x)# (B, 100, 1, T // 27)
        x = self.firstGlobalLayer(x)  # (B, 200, 1, T // 81)
        x = torch.squeeze(x, dim=2)
        x = self.tmpdropout(x)
        x_reshaped = x.permute(2, 0, 1)  # (T//81, B, 200)
        x_tempo, _ = self.temporalGlobalLayer(x_reshaped, x_reshaped, x_reshaped)
        x_tempo = x_tempo.permute(1, 2, 0)  # (B, 200, T//81)，恢复原形
        x = self.tmpdropout(x)
        x = torch.unsqueeze(torch.cat((x, x_tempo), dim=2), dim=1)# (B, 1, 200, T *  2 // 81)

        x = x.permute(0, 2, 1, 3)
        x = self.temporalLastLayer(x)


        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        loss = self.loss_fn(x, Y)
        if not self.training:
            return loss.item(), x.to(torch.float32)
        with torch.amp.autocast('cuda', enabled=False):
            if 'r2' in metrics:
                results = get_metrics(x.to(torch.float32).detach().cpu().numpy(), Y.cpu().numpy(), metrics, is_binary=False)
        log = {}
        split="train" if self.training else "val"
        log[f'{split}/loss_total'] = loss.item()
        for key, value in results.items():
            log[f'{split}/{key}'] = value

        return loss, log

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

class GaitCNN_tmp(nn.Module):
    """
    Modified GaitGraph with GraphNet block (GCN).
    """

    def __init__(self, config):
        super(GaitCNN_tmp, self).__init__()
        self.config = config.eegnet
        nChan = self.config.num_chan_eeg
        nTime = self.config.eeg.time_step
        pool_width = 3
        poolSize = {
            "LocalLayers": [(1, pool_width), (1, pool_width), (1, pool_width)],
            "GlobalLayers": (1, pool_width),
        }
        kernel_width = self.config.eeg.time_step // 20
        localKernalSize = {
            "LocalLayers": [(1, kernel_width), (1, kernel_width), (1, kernel_width)],
            "GlobalLayers": (1, kernel_width),
        }
        nClass = config.eegnet.num_chan_kin
        dropoutP = 0.5
        nFilt_FirstLayer = 25
        self.nFilt_FirstLayer = nFilt_FirstLayer
        nFiltLaterLayer = [25, 50, 100, 200]

        self.firstLayer = nn.Sequential(

            nn.ZeroPad2d((localKernalSize["LocalLayers"][0][1] // 2 - 1, localKernalSize["LocalLayers"][0][1] // 2, 0, 0)),
            Conv2dWithConstraint(
                1, nFilt_FirstLayer, localKernalSize["LocalLayers"][0], padding=0, max_norm=2),

        )

        graph2token = 'Flatten'
        encoder_type = 'GCN'
        layers_graph = [1, 2]
        K = 4
        
        self.spatialLayer = nn.Sequential(
            Conv2dWithConstraint(25, 25, (nChan, 1), padding=0, bias=False, max_norm=2),
            nn.BatchNorm2d(nFilt_FirstLayer),
            nn.ELU(),
            nn.MaxPool2d(poolSize["LocalLayers"][0], stride=poolSize["LocalLayers"][0]),
        )
        self.middleLayers = nn.Sequential(
            *[
                self.convBlock(inF, outF, dropoutP, kernalS, poolS)
                for inF, outF, kernalS, poolS in zip(
                    nFiltLaterLayer[:-1],
                    nFiltLaterLayer[1:-1],
                    localKernalSize["LocalLayers"][1:],
                    poolSize["LocalLayers"][1:],
                )
            ]
        )
        
        self.tmpdropout = nn.Dropout(p=dropoutP)
        self.firstGlobalLayer = self.convBlock(
            nFiltLaterLayer[-2],
            nFiltLaterLayer[-1],
            dropoutP,
            localKernalSize["GlobalLayers"],
            poolSize["GlobalLayers"],
        )

        # self.allButLastLayers = nn.Sequential(
        #     self.firstLayer, self.middleLayers, self.firstGlobalLayer
        # )

        # self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], nClass, (1, nTime // 81))


        # self.weight_keys = [
        #     [
        #         "allButLastLayers.0.0.weight",
        #         "allButLastLayers.0.0.bias",
        #         "allButLastLayers.0.1.weight",
        #     ],
        #     ["allButLastLayers.1.0.1.weight"],
        #     ["allButLastLayers.1.1.1.weight"],
        #     ["allButLastLayers.2.1.weight"],
        #     ["lastLayer.0.weight", "lastLayer.0.bias"],
        # ]
        self.temporalGlobalLayer = nn.MultiheadAttention(embed_dim=200, num_heads=4)

        self.temporalLastLayer = Conv2dWithConstraint(nFiltLaterLayer[-1], nClass, (1, nTime // 81 * 2), padding=0, max_norm=2)
        if self.config.loss_name == 'mse':
            self.loss_fn = nn.MSELoss()
        elif self.config.loss_name == 'freq_mse':
            self.loss_fn = FreDFLoss()
        elif self.config.loss_name == 'reward_mse':
            self.loss_fn = MSEWithEncouragingLoss()
        elif self.config.loss_name == 'freq_reward_mse':
            self.loss_fn = FreDFLossWithEncouraging()

    def convBlock(self, inF, outF, dropoutP, kernalSize, poolSize, *args, **kwargs):
        return nn.Sequential(
            nn.Dropout(p=dropoutP),
            nn.ZeroPad2d((kernalSize[1] // 2 - 1, kernalSize[1] // 2, 0, 0)),
            Conv2dWithConstraint(
                inF, outF, kernalSize, bias=False, max_norm=2, *args, **kwargs
            ),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d(poolSize, stride=poolSize),
        )


    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(

            # nn.ZeroPad2d((kernalSize[1] // 2 - 1, kernalSize[1] // 2, 0, 0)),
            Conv2dWithConstraint(inF, outF, kernalSize, max_norm=0.5, *args, **kwargs),
            # nn.LogSoftmax(dim=1),
        )

    def calculateOutSize(self, model, nChan, nTime):
        """
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        """
        data = torch.rand(1, 1, nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def forward(self, batch, metrics=None):
        # x = self.allButLastLayers(x)
        x = batch['EEG'] #(B, 1, 59, 400)
        Y = batch['Y']
        x = self.firstLayer(x) # (B, 25, C, T)
        
        x = self.spatialLayer(x)# (B, 25, 1, T // 3)

        x = self.middleLayers(x)# (B, 100, 1, T // 27)
        x = self.firstGlobalLayer(x)  # (B, 200, 1, T // 81)
        x = torch.squeeze(x, dim=2)
        x = self.tmpdropout(x)
        x_reshaped = x.permute(2, 0, 1)  # (T//81, B, 200)
        x_tempo, _ = self.temporalGlobalLayer(x_reshaped, x_reshaped, x_reshaped)
        x_tempo = x_tempo.permute(1, 2, 0)  # (B, 200, T//81)，恢复原形
        x = self.tmpdropout(x)
        x = torch.unsqueeze(torch.cat((x, x_tempo), dim=2), dim=1)# (B, 1, 200, T *  2 // 81)

        x = x.permute(0, 2, 1, 3)
        x = self.temporalLastLayer(x)


        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        loss = self.loss_fn(x, Y)
        if not self.training:
            return loss.item(), x.to(torch.float32)
        with torch.amp.autocast('cuda', enabled=False):
            if 'r2' in metrics:
                results = get_metrics(x.to(torch.float32).detach().cpu().numpy(), Y.cpu().numpy(), metrics, is_binary=False)
        log = {}
        split="train" if self.training else "val"
        log[f'{split}/loss_total'] = loss.item()
        for key, value in results.items():
            log[f'{split}/{key}'] = value

        return loss, log

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

if __name__ == "__main__":
    # 当脚本直接执行时，以下代码会运行
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x = torch.rand(100, 1, 59, 200).to(device)
    # Channel names
    ch_names = [
        'Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
        'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2',
        'Cz', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3',
        'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1',
        'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4',
        'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2'
    ]

    # Neighbors as a dictionary
    channel_neighbors = {
        # order: up-down-left-right
        "Fp1": ["AF7", "AF3", "AFz"],
        "Fp2": ["AF4", "AF8", "AFz"],
        "AF7": ["Fp1", "AF3", "F7"],
        "AF3": ["Fp1", "F3", "AF7", "AFz"],
        "AFz": ["Fp1", 'Fp2', "Fz", "AF3", "AF4"],
        "AF4": ["Fp2", "F4", "AFz", "AF8"],
        "AF8": ["Fp2", "AF4", "F8"],
        "F7": ["AF7", "FT7", "F5"],
        "F5": ["AF7", "AF3", "FC5", "F7", "F3"],
        "F3": ["AF3", "FC3", "F5", "F1"],
        "F1": ["AF3", "AFz", "FC1", "F3", "Fz"],
        "Fz": ["AFz", "F1", "F2"],
        "F2": ["AFz", "AF4", "FC2", "Fz", "F4"],
        "F4": ["AF4", "FC4", "F2", "F6"],
        "F6": ["AF4", "AF8", "FC6", "F4", "F8"],
        "F8": ["AF8", "FT8", "F6"],
        "FT7": ["F7", "T7", "FC5"],
        "FC5": ["F5", "C5", "FT7", "FC3"],
        "FC3": ["F3", "C3", "FC5", "FC1"],
        "FC1": ["F1", "C1", "FC3"],
        "FC2": ["F2", "C2", "FC4"],
        "FC4": ["F4", "C4", "FC2", "FC6"],
        "FC6": ["F6", "C6", "FC4", "FT8"],
        "FT8": ["F8", "T8", "FC6"],
        "T7": ["FT7", "TP7", "C5"],
        "C5": ["FC5", "CP5", "T7", "C3"],
        "C3": ["FC3", "CP3", "C5", "C1"],
        "C1": ["FC1", "CP1", "C3", "Cz"],
        "Cz": ["CPz", "C1", "C2"],
        "C2": ["FC2", "CP2", "Cz", "C4"],
        "C4": ["FC4", "CP4", "C2", "CP6"],
        "C6": ["FC6", "CP6", "C4", "T8"],
        "T8": ["FT8", "TP8", "C6"],
        "TP7": ["T7", "P7", "CP5"],
        "CP5": ["C5", "P5", "TP7", "CP3"],
        "CP3": ["C3", "P3", "CP5", "CP1"],
        "CP1": ["C1", "P1", "CP3", "CPz"],
        "CPz": ["Cz", "Pz", "CP1", "CP2"],
        "CP2": ["C2", "P2", "CPz", "CP4"],
        "CP4": ["C4", "P4", "CP2", "CP6"],
        "CP6": ["C6", "P6", "CP6", "TP8"],
        "TP8": ["T8", "P8", "CP6"],
        "P7": ["TP7", "PO7", "P5"],
        "P5": ["CP5", "PO7", "PO3", "P7", "P3"],
        "P3": ["CP3", "PO3", "P5", "P1"],
        "P1": ["CP1", "PO3", "POz", "P3", "Pz"],
        "Pz": ["CPz", "POz", "P1", "P2"],
        "P2": ["CP2", "POz", "PO4", "Pz", "P4"],
        "P4": ["CP4", "PO4", "P2", "P6"],
        "P6": ["CP6", "PO4", "PO8", "P4", "P8"],
        "P8": ["TP8", "PO8", "P6"],
        "PO7": ["P7", "O1", "PO3"],
        "PO3": ["P3", "O1", "Oz", "PO7", "POz"],
        "POz": ["Pz", "Oz", "PO3", "PO4"],
        "PO4": ["P4", "Oz", "O2", "POz", "PO8"],
        "PO8": ["P8", "O2", "PO4"],
        "O1": ["PO7", "PO3", "Oz"],
        "Oz": ["POz", "O1", "O2"],
        "O2": ["PO8", "PO4", "Oz"],
    }

    # # Create edge_index
    # edges = []
    #
    # # Loop through each channel and its neighbors to create edges
    # for channel, neighbors in channel_neighbors.items():
    #     for neighbor in neighbors:
    #         edges.append((ch_names.index(channel), ch_names.index(neighbor)))
    #
    # # Convert edges to tensor and create edge_index
    # edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 初始化邻接矩阵
    num_channels = len(ch_names)
    adj_matrix = np.zeros((num_channels, num_channels))

    # 填充邻接矩阵
    for channel, neighbors in channel_neighbors.items():
        channel_index = ch_names.index(channel)
        for neighbor in neighbors:
            neighbor_index = ch_names.index(neighbor)
            adj_matrix[channel_index, neighbor_index] = 1
            adj_matrix[neighbor_index, channel_index] = 1  # 无向图，双向连接

    # 将邻接矩阵转换为tensor
    adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float).to(device)

    config = load_config('../config.yaml')
    config.eegnet.adj_matrix = adj_matrix_tensor
    model = GaitGraph(config).to(device)
    out = model(x)
    a = 0