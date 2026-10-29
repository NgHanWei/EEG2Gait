import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.FreDF import FreDFLoss, FreDFLossWithEncouraging, MSEWithEncouragingLoss, GradientNormalizationLoss, FreqGradientLoss

from einops import rearrange
from einops.layers.torch import Rearrange
from utils.utils import get_metrics
import inspect


class PatchSTEmbedding(nn.Module):
    def __init__(self, emb_size, n_channels=16):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(n_channels, 64, 15, 8),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 256, 15, 8),
            Rearrange("b c s -> b s c"),
        )

    def forward(self, x):
        x = self.projection(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, sequence_num=2000, inter=100, n_channels=16):
        super(ChannelAttention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(
            self.sequence_num / self.inter
        )  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            nn.LayerNorm(
                n_channels
            ),  # also may introduce improvement to a certain extent
            nn.Dropout(0.3),
        )
        self.key = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            # nn.LeakyReLU(),
            nn.LayerNorm(n_channels),
            nn.Dropout(0.3),
        )

        # self.value = self.key
        self.projection = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            # nn.LeakyReLU(),
            nn.LayerNorm(n_channels),
            nn.Dropout(0.3),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        temp = rearrange(x, "b c s->b s c")
        temp_query = rearrange(self.query(temp), "b s c -> b c s")
        temp_key = rearrange(self.key(temp), "b s c -> b c s")

        channel_query = self.pooling(temp_query)
        channel_key = self.pooling(temp_key)

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = (
            torch.einsum("b c s, b m s -> b c m", channel_query, channel_key) / scaling
        )

        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum("b c s, b c m -> b c s", x, channel_atten_score)
        """
        projections after or before multiplying with attention score are almost the same.
        """
        out = rearrange(out, "b c s -> b s c")
        out = self.projection(out)
        out = rearrange(out, "b s c -> b c s")
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum(
            "bhqd, bhkd -> bhqk", queries, keys
        )  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input):
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self, emb_size, num_heads=8, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                    ),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class STTransformer(nn.Module):
    """
    Refer to https://arxiv.org/abs/2106.11170
    Modified from https://github.com/eeyhsong/EEG-Transformer
    """

    def __init__(
        self,
        emb_size=256,
        depth=3,
        n_classes=4,
        channel_legnth=2000,
        n_channels=16,
        loss_name='mse',
        **kwargs
    ):
        super().__init__()
        self.channel_attension = ResidualAdd(
            nn.Sequential(
                nn.LayerNorm(channel_legnth),
                ChannelAttention(n_channels=n_channels),
                nn.Dropout(0.5),
            )
        )
        self.patch_embedding = PatchSTEmbedding(emb_size, n_channels)
        self.transformer = TransformerEncoder(depth, emb_size)
        self.classification = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, n_classes),
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
        x = self.channel_attension(x)
        x = self.patch_embedding(x)
        x = self.transformer(x).mean(dim=1)
        x = self.classification(x)
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
    model = STTransformer(n_classes=6)
    out = model(X)
    print(out.shape)
