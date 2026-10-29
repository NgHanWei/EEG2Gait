import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils.FreDF import FreDFLoss, FreDFLossWithEncouraging, MSEWithEncouragingLoss, GradientNormalizationLoss, FreqGradientLoss


from utils.utils import get_metrics
import inspect

# refer to https://github.com/ravikiran-mane/FBCNet/blob/master/codes/centralRepo/networks.py
class deepConvNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        nChan = config.num_chan_eeg
        nTime = config.eegnet.eeg.time_step
        print('nChan',nChan)
        print('nTime',nTime)
        pool_width = 3
        poolSize = {
                "LocalLayers": [(1, pool_width), (1, pool_width), (1, pool_width)],
                "GlobalLayers": (1, pool_width),
            }
        kernel_width = 20
        localKernalSize = {
                "LocalLayers": [(1, kernel_width), (1, kernel_width), (1, kernel_width)],
                "GlobalLayers": (1, kernel_width),
            }
        nClass = config.eegnet.num_chan_kin
        dropoutP = 0.5
        nFilt_FirstLayer = 25
        nFiltLaterLayer = [25, 50, 100, 200]

        self.firstLayer = self.firstBlock(
            nFilt_FirstLayer,
            dropoutP,
            localKernalSize["LocalLayers"][0],
            nChan,
            poolSize["LocalLayers"][0],
        )
        # middleLayers = nn.Sequential(*[self.convBlock(inF, outF, dropoutP, localKernalSize)
        #     for inF, outF in zip(nFiltLaterLayer[:-1], nFiltLaterLayer[1:-1])])
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
        self.firstGlobalLayer = self.convBlock(
            nFiltLaterLayer[-2],
            nFiltLaterLayer[-1],
            dropoutP,
            localKernalSize["GlobalLayers"],
            poolSize["GlobalLayers"],
        )

        self.allButLastLayers = nn.Sequential(
            self.firstLayer, self.middleLayers, self.firstGlobalLayer
        )

        self.fSize = self.calculateOutSize(self.allButLastLayers, nChan, nTime)
        print('self.fSize',self.fSize)
        self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], nClass, (1, self.fSize[1]))

        # self.weight_keys =[['allButLastLayers.0.0.weight','allButLastLayers.0.0.bias','allButLastLayers.0.1.weight'],
        #                     ['allButLastLayers.1.0.1.weight'],
        #                     ['allButLastLayers.1.1.1.weight'],
        #                     ['allButLastLayers.1.2.1.weight'],
        #                     ['lastLayer.0.weight', 'lastLayer.0.bias']
        #                     ]

        self.weight_keys = [
            [
                "allButLastLayers.0.0.weight",
                "allButLastLayers.0.0.bias",
                "allButLastLayers.0.1.weight",
            ],
            ["allButLastLayers.1.0.1.weight"],
            ["allButLastLayers.1.1.1.weight"],
            ["allButLastLayers.2.1.weight"],
            ["lastLayer.0.weight", "lastLayer.0.bias"],
        ]
        if config.eegnet.loss_name == 'mse':
            self.loss_fn = nn.MSELoss()
        elif config.eegnet.loss_name == 'freq_mse':
            self.loss_fn = FreDFLoss()
        elif config.eegnet.loss_name == 'reward_mse':
            self.loss_fn = MSEWithEncouragingLoss()
        elif config.eegnet.loss_name == 'freq_reward_mse':
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

    def firstBlock(self, outF, dropoutP, kernalSize, nChan, poolSize, *args, **kwargs):
        return nn.Sequential(

            nn.ZeroPad2d((kernalSize[1] // 2 - 1, kernalSize[1] // 2, 0, 0)),
            Conv2dWithConstraint(
                1, outF, kernalSize, padding=0, max_norm=2, *args, **kwargs
            ),
            Conv2dWithConstraint(outF, outF, (nChan, 1), padding=0, bias=False, max_norm=2),
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
        x = batch['EEG'] #(B, 1, 59, 400)
        Y = batch['Y']
        x = self.firstLayer(x) #(B, 25, 1, 400//3)
        
        x = self.middleLayers(x) #(B, 100, 1, 14)
        x = self.firstGlobalLayer(x) #(B, 200, 1, 400//81)
        x = self.lastLayer(x) #(B, 6, 1, 1)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2) #(B, 6)

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

class deepConvNet_multihead(nn.Module):

    def __init__(self, config, n_domains=100, target_domains = [98, 99]):
        super().__init__()
        nChan = config.num_chan_eeg
        nTime = config.eegnet.eeg.time_step
        print('nChan',nChan)
        print('nTime',nTime)
        self.n_domains = n_domains
        pool_width = 3
        poolSize = {
                "LocalLayers": [(1, pool_width), (1, pool_width), (1, pool_width)],
                "GlobalLayers": (1, pool_width),
            }
        kernel_width = 20
        localKernalSize = {
                "LocalLayers": [(1, kernel_width), (1, kernel_width), (1, kernel_width)],
                "GlobalLayers": (1, kernel_width),
            }
        nClass = config.eegnet.num_chan_kin
        dropoutP = 0.5
        nFilt_FirstLayer = 25
        nFiltLaterLayer = [25, 50, 100, 200]
        
        self.target_domains = target_domains

        self.firstLayer = self.firstBlock(
            nFilt_FirstLayer,
            dropoutP,
            localKernalSize["LocalLayers"][0],
            nChan,
            poolSize["LocalLayers"][0],
        )
        # middleLayers = nn.Sequential(*[self.convBlock(inF, outF, dropoutP, localKernalSize)
        #     for inF, outF in zip(nFiltLaterLayer[:-1], nFiltLaterLayer[1:-1])])
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
        self.firstGlobalLayer = self.convBlock(
            nFiltLaterLayer[-2],
            nFiltLaterLayer[-1],
            dropoutP,
            localKernalSize["GlobalLayers"],
            poolSize["GlobalLayers"],
        )

        self.allButLastLayers = nn.Sequential(
            self.firstLayer, self.middleLayers, self.firstGlobalLayer
        )

        self.fSize = self.calculateOutSize(self.allButLastLayers, nChan, nTime)
        print('self.fSize',self.fSize)
        self.domain_conv = nn.Conv1d(in_channels=nTime // 81, out_channels=1, kernel_size=1)
        self.domain_linear = nn.Linear(nFiltLaterLayer[-1], self.n_domains)

        self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], nClass * self.n_domains, (1, self.fSize[1]))
        self.n_classes = nClass

        # self.weight_keys =[['allButLastLayers.0.0.weight','allButLastLayers.0.0.bias','allButLastLayers.0.1.weight'],
        #                     ['allButLastLayers.1.0.1.weight'],
        #                     ['allButLastLayers.1.1.1.weight'],
        #                     ['allButLastLayers.1.2.1.weight'],
        #                     ['lastLayer.0.weight', 'lastLayer.0.bias']
        #                     ]

        self.weight_keys = [
            [
                "allButLastLayers.0.0.weight",
                "allButLastLayers.0.0.bias",
                "allButLastLayers.0.1.weight",
            ],
            ["allButLastLayers.1.0.1.weight"],
            ["allButLastLayers.1.1.1.weight"],
            ["allButLastLayers.2.1.weight"],
            ["lastLayer.0.weight", "lastLayer.0.bias"],
        ]
        self.loss_fn = nn.MSELoss()
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

    def firstBlock(self, outF, dropoutP, kernalSize, nChan, poolSize, *args, **kwargs):
        return nn.Sequential(

            nn.ZeroPad2d((kernalSize[1] // 2 - 1, kernalSize[1] // 2, 0, 0)),
            Conv2dWithConstraint(
                1, outF, kernalSize, padding=0, max_norm=2, *args, **kwargs
            ),
            Conv2dWithConstraint(outF, outF, (nChan, 1), padding=0, bias=False, max_norm=2),
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
        x = batch['EEG'] #(B, 1, 59, 400)
        Y = batch['Y']
        domain = batch['domain']  # (B,)
        x = self.firstLayer(x) #(B, 25, 1, 400//3)
        x = self.middleLayers(x) #(B, 100, 1, 14)
        x = self.firstGlobalLayer(x) #(B, 200, 1, 400//81)
        
        domain_weights = self.domain_linear(self.domain_conv(x.squeeze().permute(0, 2, 1)).squeeze()) #(B, n_domains)
        domain_weights[torch.arange(x.size(0)), domain] = -float('inf')  # (B, 100)
        domain_weights[:, self.target_domains] = -float('inf')# (B, 100)
        domain_weights = torch.softmax(domain_weights, dim=-1)  # 归一化，使其总和为 1
        # 计算 cross-domain 的加权和
        domain_weights = domain_weights.unsqueeze(-1) #(B, n_domains, 1)
        
        x = self.lastLayer(x) # (B, n_classes*n_domains, 1, 1)
        logits_interdomain = x.view(x.shape[0], self.n_domains, self.n_classes)        
        
        
        logits_cross = torch.sum(logits_interdomain * domain_weights, dim=1)  # (B, n_classes)
        # 计算一致性损失 (L_rel)
        loss_cross = self.loss_fn(logits_cross, Y)  # 让 cross-domain 预测接近 intra-domain

        # 评估模式
        if not self.training:
            return loss_cross.item(), logits_cross.to(torch.float32)
        # 获取当前 domain head 输出 (Intra-domain prediction)
        logits_intra = logits_interdomain[torch.arange(x.size(0)), domain]  # (B, n_classes)
        # 计算监督损失 (L_pred)
        loss_intra = self.loss_fn(logits_intra, Y)

        
        loss_total = loss_intra + 0.2 * loss_cross


        with torch.amp.autocast('cuda', enabled=False):
            if 'r2' in metrics:
                results = get_metrics(logits_intra.to(torch.float32).detach().cpu().numpy(), 
                                      Y.cpu().numpy(), metrics, is_binary=False)

        log = {}
        split="train" if self.training else "val"
        log[f'{split}/loss_total'] = loss_total.item()
        for key, value in results.items():
            log[f'{split}/{key}'] = value

        return loss_total, log


class deepConvNet_latespatial(nn.Module):

    def __init__(self, config):
        super().__init__()
        nChan = config.num_chan_eeg
        nTime = config.eegnet.eeg.time_step
        print('nChan',nChan)
        print('nTime',nTime)
        pool_width = 3
        poolSize = {
                "LocalLayers": [(1, pool_width), (1, pool_width), (1, pool_width)],
                "GlobalLayers": (1, pool_width),
            }
        kernel_width = 20
        localKernalSize = {
                "LocalLayers": [(1, kernel_width), (1, kernel_width), (1, kernel_width)],
                "GlobalLayers": (1, kernel_width),
            }
        nClass = config.eegnet.num_chan_kin
        dropoutP = 0.5
        nFilt_FirstLayer = 25
        nFiltLaterLayer = [25, 50, 100, 200]

        self.firstLayer = self.firstBlock(
            nFilt_FirstLayer,
            dropoutP,
            localKernalSize["LocalLayers"][0],
            nChan,
            poolSize["LocalLayers"][0],
        )
        # middleLayers = nn.Sequential(*[self.convBlock(inF, outF, dropoutP, localKernalSize)
        #     for inF, outF in zip(nFiltLaterLayer[:-1], nFiltLaterLayer[1:-1])])
        self.middleLayers = nn.Sequential(
            *[
                self.convBlock(inF, outF, dropoutP, kernalS, poolS)
                for inF, outF, kernalS, poolS in zip(
                    nFiltLaterLayer[:-2],
                    nFiltLaterLayer[1:-1],
                    localKernalSize["LocalLayers"][1:],
                    poolSize["LocalLayers"][1:],
                )
            ]
        )
        self.firstGlobalLayer = self.convBlock(
            nFiltLaterLayer[-2],
            nFiltLaterLayer[-1],
            dropoutP,
            localKernalSize["GlobalLayers"],
            poolSize["GlobalLayers"],
        )

        self.allButLastLayers = nn.Sequential(
            self.firstLayer, self.middleLayers, self.firstGlobalLayer
        )

        self.fSize = self.calculateOutSize(self.allButLastLayers, nChan, nTime)
        print('self.fSize',self.fSize)
        self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], nClass, (1, self.fSize[1]))

        # self.weight_keys =[['allButLastLayers.0.0.weight','allButLastLayers.0.0.bias','allButLastLayers.0.1.weight'],
        #                     ['allButLastLayers.1.0.1.weight'],
        #                     ['allButLastLayers.1.1.1.weight'],
        #                     ['allButLastLayers.1.2.1.weight'],
        #                     ['lastLayer.0.weight', 'lastLayer.0.bias']
        #                     ]

        self.weight_keys = [
            [
                "allButLastLayers.0.0.weight",
                "allButLastLayers.0.0.bias",
                "allButLastLayers.0.1.weight",
            ],
            ["allButLastLayers.1.0.1.weight"],
            ["allButLastLayers.1.1.1.weight"],
            ["allButLastLayers.2.1.weight"],
            ["lastLayer.0.weight", "lastLayer.0.bias"],
        ]
        self.loss_fn = nn.MSELoss()
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

    def firstBlock(self, outF, dropoutP, kernalSize, nChan, poolSize, *args, **kwargs):
        return nn.Sequential(

            nn.ZeroPad2d((kernalSize[1] // 2 - 1, kernalSize[1] // 2, 0, 0)),
            Conv2dWithConstraint(
                1, outF, kernalSize, padding=0, max_norm=2, *args, **kwargs
            ),
            # Conv2dWithConstraint(outF, outF, (nChan, 1), padding=0, bias=False, max_norm=2),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d(poolSize, stride=poolSize),
        )

    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(

            # nn.ZeroPad2d((kernalSize[1] // 2 - 1, kernalSize[1] // 2, 0, 0)),
            Conv2dWithConstraint(inF, inF, (59, 1), padding=0, bias=False, max_norm=2),
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
        x = batch['EEG'] #(B, 1, 59, 400)
        Y = batch['Y']
        x = self.firstLayer(x) #(B, 25, 59, 400//3)
        x = self.middleLayers(x) #(B, 100, 59, 14)
        x = self.firstGlobalLayer(x) #(B, 200, 59, 400//81)
        x = self.lastLayer(x) #(B, 6, 1, 1)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2) #(B, 6)

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

class deepConvNet_sv(nn.Module):

    def __init__(self, config):
        super().__init__()
        nChan = config.num_chan_eeg
        nTime = config.eegnet.eeg.time_step
        print('nChan',nChan)
        print('nTime',nTime)
        pool_width = 9
        poolSize = {
                "LocalLayers": [(1, pool_width)],
                "GlobalLayers": (1, pool_width),
            }
        kernel_width = 20
        localKernalSize = {
                "LocalLayers": [(1, kernel_width)],
                "GlobalLayers": (1, kernel_width),
            }
        nClass = config.eegnet.num_chan_kin
        dropoutP = 0.5
        nFilt_FirstLayer = 50
        nFiltLaterLayer = [50, 200]

        self.firstLayer = self.firstBlock(
            nFilt_FirstLayer,
            dropoutP,
            localKernalSize["LocalLayers"][0],
            nChan,
            poolSize["LocalLayers"][0],
        )
        
        self.firstGlobalLayer = self.convBlock(
            nFiltLaterLayer[-2],
            nFiltLaterLayer[-1],
            dropoutP,
            localKernalSize["GlobalLayers"],
            poolSize["GlobalLayers"],
        )

        self.allButLastLayers = nn.Sequential(
            self.firstLayer, self.firstGlobalLayer
        )

        self.fSize = self.calculateOutSize(self.allButLastLayers, nChan, nTime)
        print('self.fSize',self.fSize)
        self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], nClass, (1, self.fSize[1]))

        # self.weight_keys =[['allButLastLayers.0.0.weight','allButLastLayers.0.0.bias','allButLastLayers.0.1.weight'],
        #                     ['allButLastLayers.1.0.1.weight'],
        #                     ['allButLastLayers.1.1.1.weight'],
        #                     ['allButLastLayers.1.2.1.weight'],
        #                     ['lastLayer.0.weight', 'lastLayer.0.bias']
        #                     ]

        self.weight_keys = [
            [
                "allButLastLayers.0.0.weight",
                "allButLastLayers.0.0.bias",
                "allButLastLayers.0.1.weight",
            ],
            ["allButLastLayers.1.0.1.weight"],
            ["allButLastLayers.1.1.1.weight"],
            ["allButLastLayers.2.1.weight"],
            ["lastLayer.0.weight", "lastLayer.0.bias"],
        ]
        self.loss_fn = nn.MSELoss()
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

    def firstBlock(self, outF, dropoutP, kernalSize, nChan, poolSize, *args, **kwargs):
        return nn.Sequential(

            nn.ZeroPad2d((kernalSize[1] // 2 - 1, kernalSize[1] // 2, 0, 0)),
            Conv2dWithConstraint(
                1, outF, kernalSize, padding=0, max_norm=2, *args, **kwargs
            ),
            Conv2dWithConstraint(outF, outF, (nChan, 1), padding=0, bias=False, max_norm=2),
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
        x = batch['EEG'] #(B, 1, 59, 400)
        Y = batch['Y']
        x = self.firstLayer(x) #(B, 50, 1, 400//9)
        x = self.firstGlobalLayer(x) #(B, 200, 1, 400//81)
        x = self.lastLayer(x) #(B, 6, 1, 1)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2) #(B, 6)

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

class shallowConvNet(nn.Module):
    def convBlock(self, inF, outF, dropoutP, kernalSize, *args, **kwargs):
        return nn.Sequential(
            nn.Dropout(p=dropoutP),
            Conv2dWithConstraint(inF, outF, kernalSize, bias=False, max_norm=2, *args, **kwargs),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d((1, 3), stride=(1, 3))
        )

    def firstBlock(self, outF, dropoutP, kernalSize, nChan, *args, **kwargs):
        return nn.Sequential(
            Conv2dWithConstraint(1, outF, kernalSize, padding=0, max_norm=2, *args, **kwargs),
            Conv2dWithConstraint(40, 40, (nChan, 1), padding=0, bias=False, max_norm=2),
            nn.BatchNorm2d(outF),
        )

    def calculateOutSize(self, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1, 1, nChan, nTime)
        block_one = self.firstLayer
        avg = self.avgpool
        dp = self.dp
        out = torch.log(block_one(data).pow(2))
        out = avg(out)
        out = dp(out)
        out = out.view(out.size()[0], -1)
        return out.size()

    def __init__(self, config):
        super(shallowConvNet, self).__init__()
        nChan = config.num_chan_eeg
        nTime = config.eegnet.eeg.time_step
        nClass = 6
        dropoutP = 0.25
        kernalSize = (1, 25)
        nFilt_FirstLayer = 40

        self.firstLayer = self.firstBlock(nFilt_FirstLayer, dropoutP, kernalSize, nChan)
        self.avgpool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dp = nn.Dropout(p=dropoutP)
        self.fSize = self.calculateOutSize(nChan, nTime)
        self.lastLayer = nn.Linear(self.fSize[-1], nClass)

    def forward(self, x):
        x = self.firstLayer(x)
        x = torch.log(self.avgpool(x.pow(2)))
        x = self.dp(x)
        x = x.view(x.size()[0], -1)
        x = self.lastLayer(x)

        return x


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=False, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super().forward(x)


# 定义模型配置
class Config:
    def __init__(self):
        self.num_chan_eeg = 59
        self.eegnet = type('', (), {})()
        self.eegnet.eeg = type('', (), {})()
        self.eegnet.eeg.time_step = 400
        self.eegnet.num_chan_kin = 6


if __name__ == "__main__":
    # 初始化模型
    config = Config()
    model = deepConvNet_latespatial(config)

    # 生成随机输入数据
    batch_size = 8
    nChan = config.num_chan_eeg
    nTime = config.eegnet.eeg.time_step
    nClass = config.eegnet.num_chan_kin

    # 随机生成 EEG 数据和标签
    eeg_data = torch.randn(batch_size, 1, nChan, nTime)
    labels = torch.randn(batch_size, nClass)

    # 创建 DataLoader
    dataset = TensorDataset(eeg_data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def test_model():
        # 测试前向传播
        for batch in dataloader:
            eeg, y = batch
            batch_data = {'EEG': eeg, 'Y': y}
            
            # 前向传播
            loss, log = model(batch_data, metrics=['r2'])
            
            print(f"Loss: {loss.item()}")
            for key, value in log.items():
                print(f"{key}: {value}")

        # 测试优化器配置
        optimizer = model.configure_optimizers(weight_decay=1e-4, learning_rate=1e-3, betas=(0.9, 0.999), device_type='cuda')
        print("Optimizer configured successfully.")

        # 测试模型训练模式
        model.train()
        for batch in dataloader:
            eeg, y = batch
            batch_data = {'EEG': eeg, 'Y': y}
            
            # 前向传播和损失计算
            loss, log = model(batch_data, metrics=['r2'])
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Training Loss: {loss.item()}")

        # 测试模型评估模式
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                eeg, y = batch
                batch_data = {'EEG': eeg, 'Y': y}
                
                # 前向传播和损失计算
                loss, log = model(batch_data, metrics=['r2'])
                
                print(f"Validation Loss: {loss.item()}")

    test_model()