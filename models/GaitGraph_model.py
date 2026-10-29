import contextlib
import inspect
import math
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .DCN_model import Conv2dWithConstraint

from utils.FreDF import (
    FreDFLoss,
    FreDFLossWithEncouraging,
    MSEWithEncouragingLoss,
)

from utils.utils import get_metrics


def _make_activation(name):
    """Small factory for the activations used inside the GCN."""

    if name == "relu":
        return nn.ReLU()

    if name == "elu":
        return nn.ELU()

    if name == "none":
        return nn.Identity()

    raise ValueError(
        f"Unknown activation '{name}' (expected relu | elu | none)"
    )


# ----------------------------------------------------------------------
# Presets
#
# config.eegnet.preset selects a bundle of flag values. Any flag you set
# explicitly on config.eegnet overrides the preset.
#
#   "original"      your first GaitGraph_tmp: full spatial conv, max pool,
#                   no GSL dropout, double-processed adjacency init, weight
#                   decay on the adjacency, kernel = T // 20; every
#                   optional extra OFF.
#   "conservative"  (default) "original" + only the low-risk training
#                   recipe: cosine LR (warmup + constant until
#                   lr_decay_iters is set), grad clip 1.0, EMA 0.999.
#   "paper"         paper-faithful architecture (depthwise GSL + dropout +
#                   avg pool, raw adjacency init, no decay on the
#                   adjacency, kernel 10); extras OFF.
#   "full"          "paper" + every optional extra ON (the previous
#                   defaults of this file).
# ----------------------------------------------------------------------

_ORIGINAL_FLAGS = dict(
    # architecture
    kernel_width="auto",          # "auto" = T // 20, or an int
    gsl_depthwise=False,          # depthwise (grouped) spatial conv
    gsl_dropout=False,            # dropout in the Global Spatial Learner
    gsl_pool="max",               # "max" | "avg"
    adj_init="original",          # "original" (double-processed) | "raw"
    adj_no_decay=False,           # exclude the adjacency from weight decay
    temporal_pools=[3, 3, 3, 3],
    attn_variant="paper",
    attn_heads=4,
    attn_dropout=0.1,
    attn_ff_mult=2,
    attn_pos_enc=True,
    gcn_last_act="relu",
    gcn_norm=False,
    gcn_residual=False,
    # targets / loss
    target_norm=False,
    target_stats_auto=True,
    corr_loss_weight=0.0,
    adj_prior_weight=0.0,
    adj_prior_type="l2",
    # augmentation
    aug_time_shift=0,
    aug_noise_std=0.0,
    aug_amp_scale=0.0,
    aug_channel_drop=0.0,
    # optimisation helpers
    lr_schedule="none",
    warmup_iters=0,
    lr_decay_iters=0,
    min_lr_ratio=0.1,
    grad_clip=0.0,
    ema_decay=0.0,
)

_PAPER_ARCH = dict(
    kernel_width=10,
    gsl_depthwise=True,
    gsl_dropout=True,
    gsl_pool="avg",
    adj_init="raw",
    adj_no_decay=True,
)

_TRAIN_RECIPE = dict(
    lr_schedule="cosine",
    warmup_iters=100,
    grad_clip=1.0,
    ema_decay=0.999,
)

_PRESETS = {
    "original": dict(_ORIGINAL_FLAGS),
    "conservative": {
        **_ORIGINAL_FLAGS,
        **_TRAIN_RECIPE,
    },
    "paper": {
        **_ORIGINAL_FLAGS,
        **_PAPER_ARCH,
    },
    "full": dict(
        {
            **_ORIGINAL_FLAGS,
            **_PAPER_ARCH,
            **_TRAIN_RECIPE,
        },
        target_norm=True,
        corr_loss_weight=0.5,
        adj_prior_weight=0.05,
        aug_time_shift=5,
        aug_noise_std=0.05,
        aug_amp_scale=0.1,
        aug_channel_drop=0.1,
        temporal_pools=[3, 2, 2, 2],
        attn_variant="enhanced",
        gcn_last_act="elu",
        gcn_norm=True,
        gcn_residual=True,
    ),
}

_DEFAULT_PRESET = "conservative"


class GaitGraph_tmp(nn.Module):
    """
    EEG2GAIT-style EEG-to-gait decoder.

    Architecture
    ------------
    EEG
      |
      +-- (training only) optional augmentation
      |
      +-- Local Temporal Learner
      |
      +-- Hierarchical Graph Convolutional Pyramid
      |       |- GCN depth = 1
      |       |- GCN depth = 2
      |
      +-- residual graph fusion
      |
      +-- Global Spatial Learner
      |       (C,1) conv [optionally depthwise] -> BN -> ELU
      |       [-> Dropout] -> Max/Avg pool
      |
      +-- Feature Fusion Layers
      |       25 -> 50 -> 100 -> 200
      |
      +-- Global Temporal Learner
      |       Multi-head self-attention + residual
      |       (optional: pre-norm + FFN + positional encoding)
      |
      +-- Output layer
      |
      +-- gait/joint prediction

    Expected input:
        batch["EEG"]: (B, 1, C, T)

    Expected target:
        batch["Y"]:   (B, num_chan_kin)

    ------------------------------------------------------------------
    Presets and flags (all read from config.eegnet)
    ------------------------------------------------------------------

    config.eegnet.preset picks a bundle of flag values (default
    "conservative"); any flag set explicitly on config.eegnet overrides
    the preset. See _PRESETS at the top of the file for the exact
    values:

        "original"      your first version, every extra off
        "conservative"  "original" + cosine LR, grad clip, EMA   (default)
        "paper"         paper-faithful architecture, extras off
        "full"          "paper" + every extra on

    Architecture
        kernel_width       int|"auto"   temporal kernel; "auto" = T // 20
        gsl_depthwise      bool   depthwise (grouped) spatial conv
        gsl_dropout        bool   dropout (0.5) inside the GSL
        gsl_pool           str    "max" | "avg"
        adj_init           str    "original" (ReLU(A+A^T)+I, then get_adj
                                  repeats it) | "raw" (paper: A only)
        adj_no_decay       bool   exclude the adjacency from weight decay
        temporal_pools     list   pools for [GSL, block1, block2, block3];
                                  [3,3,3,3] = paper, [3,2,2,2] -> 8 tokens
                                  at T=200
        attn_variant       str    "paper" | "enhanced"
        attn_heads         int
        attn_dropout       float  (enhanced variant only)
        attn_ff_mult       int    (enhanced variant only)
        attn_pos_enc       bool   (enhanced variant only)
        gcn_last_act       str    "relu" | "elu" | "none"
        gcn_norm           bool   LayerNorm inside each GCN layer
        gcn_residual       bool   residual across the depth-2 branch

    Targets / loss
        target_norm        bool   z-score targets (call
                                  fit_target_stats()/set_target_stats())
        target_stats_auto  bool   fall back to first-batch statistics
        corr_loss_weight   float  weight of (1 - Pearson r)
        adj_prior_weight   float  weight of ||A - A_init||
        adj_prior_type     str    "l2" | "l1"

    Augmentation (training only; 0 = off)
        aug_time_shift     int    max +-shift in samples
        aug_noise_std      float  noise, fraction of per-channel std
        aug_amp_scale      float  random gain in [1-a, 1+a]
        aug_channel_drop   float  per-channel drop probability

    Optimization helpers (call from your training loop, see methods)
        lr_schedule        str    "none" | "cosine"
        warmup_iters       int
        lr_decay_iters     int    set to total iters to enable decay
        min_lr_ratio       float
        grad_clip          float  0 = off
        ema_decay          float  0 = off
    """

    # ============================================================
    # Small internal GCN layer
    # ============================================================

    class _GraphConv(nn.Module):
        """
        One graph convolution:

            X' = act( norm( A_hat X W + b ) )

        Input:
            x   : (B, N, Fin)
            adj : (N, N)

        Output:
            x   : (B, N, Fout)
        """

        def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            act="relu",
            use_norm=False,
        ):
            super().__init__()

            self.linear = nn.Linear(
                in_features,
                out_features,
                bias=bias,
            )

            # NEW (flag: gcn_norm): LayerNorm over the feature dim.
            self.norm = (
                nn.LayerNorm(out_features)
                if use_norm
                else nn.Identity()
            )

            self.act = _make_activation(act)

        def forward(self, x, adj):

            # Graph propagation
            #
            # (N,N) @ (B,N,F)
            #
            x = torch.matmul(
                adj,
                x,
            )

            # Feature projection
            x = self.linear(x)

            x = self.norm(x)

            x = self.act(x)

            return x

    # ============================================================
    # Hierarchical GCN
    # ============================================================

    class _HierarchicalGCN(nn.Module):
        """
        GCN branch with configurable depth.

        depth=1:
            input -> GCN

        depth=2:
            input -> GCN -> GCN

        Hidden layers always use ReLU; the last layer uses `last_act`.
        """

        def __init__(
            self,
            in_features,
            hidden_features,
            out_features,
            depth,
            last_act="relu",
            use_norm=False,
            residual=False,
        ):
            super().__init__()

            if depth < 1:
                raise ValueError(
                    "GCN depth must be >= 1"
                )

            layers = []

            if depth == 1:

                layers.append(
                    GaitGraph_tmp._GraphConv(
                        in_features,
                        out_features,
                        act=last_act,
                        use_norm=use_norm,
                    )
                )

            else:

                layers.append(
                    GaitGraph_tmp._GraphConv(
                        in_features,
                        hidden_features,
                        act="relu",
                        use_norm=use_norm,
                    )
                )

                for _ in range(depth - 2):

                    layers.append(
                        GaitGraph_tmp._GraphConv(
                            hidden_features,
                            hidden_features,
                            act="relu",
                            use_norm=use_norm,
                        )
                    )

                layers.append(
                    GaitGraph_tmp._GraphConv(
                        hidden_features,
                        out_features,
                        act=last_act,
                        use_norm=use_norm,
                    )
                )

            self.layers = nn.ModuleList(
                layers
            )

            # NEW (flag: gcn_residual): only meaningful for depth >= 2
            # (the depth-1 branch already gets the outer residual).
            self.residual = (
                residual
                and depth >= 2
                and in_features == out_features
            )

        def forward(
            self,
            x,
            adj,
        ):

            identity = x

            for layer in self.layers:
                x = layer(x, adj)

            if self.residual:
                x = x + identity

            return x

    # ============================================================
    # Global temporal attention (enhanced variant)
    # ============================================================

    class _TemporalAttentionBlock(nn.Module):
        """
        NEW (flag: attn_variant="enhanced").

        Pre-norm Transformer block with optional sinusoidal positional
        encoding, attention dropout and a small feed-forward network.
        The residual connections are inside the block.

        Input / output: (B, D, L)
        """

        def __init__(
            self,
            d_model,
            num_heads,
            dropout,
            ff_mult,
            max_len,
            use_pos,
        ):
            super().__init__()

            self.use_pos = use_pos

            self.norm1 = nn.LayerNorm(d_model)

            self.attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

            self.norm2 = nn.LayerNorm(d_model)

            self.ff = nn.Sequential(
                nn.Linear(d_model, d_model * ff_mult),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * ff_mult, d_model),
            )

            self.drop = nn.Dropout(dropout)

            if use_pos:

                position = torch.arange(
                    max_len,
                    dtype=torch.float32,
                ).unsqueeze(1)

                div_term = torch.exp(
                    torch.arange(
                        0,
                        d_model,
                        2,
                        dtype=torch.float32,
                    )
                    * (-math.log(10000.0) / d_model)
                )

                pe = torch.zeros(
                    max_len,
                    d_model,
                )

                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(
                    position * div_term[: d_model // 2]
                )

                self.register_buffer(
                    "pos",
                    pe,
                    persistent=False,
                )

        def forward(self, x):

            # (B, D, L) -> (B, L, D)
            t = x.transpose(1, 2)

            if self.use_pos:
                t = t + self.pos[: t.size(1)].unsqueeze(0)

            h = self.norm1(t)

            a, _ = self.attn(
                h,
                h,
                h,
                need_weights=False,
            )

            t = t + self.drop(a)

            h = self.norm2(t)

            t = t + self.drop(self.ff(h))

            # (B, L, D) -> (B, D, L)
            return t.transpose(1, 2)

    # ============================================================
    # Config helper
    # ============================================================

    def _raw_cfg(self, name):
        """
        Value of `name` set explicitly on config.eegnet, or None when it
        is not set.
        """

        try:
            value = getattr(self.config, name)
        except (AttributeError, KeyError):
            return None

        if value is None:
            return None

        # DotMap (and similar auto-vivifying config objects) return an
        # EMPTY mapping for a missing key instead of raising
        # AttributeError. None of the flags is ever a mapping, so an
        # empty mapping means "not set".
        if isinstance(value, Mapping) and len(value) == 0:
            return None

        if type(value).__name__ == "DotMap" and len(value) == 0:
            return None

        return value

    def _cfg(self, name, default):
        """
        Resolve a flag. Precedence:
            1. explicit value on config.eegnet
            2. the selected preset
            3. `default`
        """

        value = self._raw_cfg(name)

        if value is not None:
            return value

        preset = getattr(self, "_preset_values", None)

        if preset is not None and name in preset:
            return preset[name]

        return default

    # ============================================================
    # Constructor
    # ============================================================

    def __init__(
        self,
        config,
        init_adj=True,
    ):
        super(GaitGraph_tmp, self).__init__()

        # --------------------------------------------------------
        # Configuration
        # --------------------------------------------------------

        self.config = config.eegnet

        nChan = self.config.num_chan_eeg
        nTime = self.config.eeg.time_step
        nClass = self.config.num_chan_kin

        self.nChan = nChan
        self.nTime = nTime
        self.nClass = nClass

        print(
            f"EEG2GAIT: nChan={nChan}, nTime={nTime}, "
            f"nClass={nClass}"
        )

        # --------------------------------------------------------
        # Preset
        # --------------------------------------------------------

        preset_name = str(
            self._raw_cfg("preset") or _DEFAULT_PRESET
        )

        if preset_name not in _PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_name}' "
                f"(expected one of {sorted(_PRESETS)})"
            )

        self.preset_name = preset_name
        self._preset_values = _PRESETS[preset_name]

        # Effective value of every flag (explicit config > preset).
        self.effective_flags = {
            name: self._cfg(name, default)
            for name, default in _ORIGINAL_FLAGS.items()
        }

        print(
            f"EEG2GAIT preset='{preset_name}' "
            f"(explicit config values override it)"
        )

        for name in sorted(self.effective_flags):
            print(f"    {name} = {self.effective_flags[name]}")

        # --------------------------------------------------------
        # Optional-feature flags
        # --------------------------------------------------------

        # Targets / loss
        self.target_norm = bool(self._cfg("target_norm", False))
        self.corr_loss_weight = float(self._cfg("corr_loss_weight", 0.0))
        self.adj_prior_weight = float(self._cfg("adj_prior_weight", 0.0))
        self.adj_prior_type = str(self._cfg("adj_prior_type", "l2"))
        self.adj_no_decay = bool(self._cfg("adj_no_decay", False))

        if self.adj_prior_type not in ("l1", "l2"):
            raise ValueError(
                "adj_prior_type must be 'l1' or 'l2'"
            )

        # Augmentation
        self.aug_time_shift = int(self._cfg("aug_time_shift", 0))
        self.aug_noise_std = float(self._cfg("aug_noise_std", 0.0))
        self.aug_amp_scale = float(self._cfg("aug_amp_scale", 0.0))
        self.aug_channel_drop = float(self._cfg("aug_channel_drop", 0.0))

        # Optimisation helpers
        self.lr_schedule = str(self._cfg("lr_schedule", "none"))
        self.warmup_iters = int(self._cfg("warmup_iters", 0))
        self.lr_decay_iters = int(self._cfg("lr_decay_iters", 0))
        self.min_lr_ratio = float(self._cfg("min_lr_ratio", 0.1))
        self.grad_clip = float(self._cfg("grad_clip", 0.0))
        self.ema_decay = float(self._cfg("ema_decay", 0.0))

        if self.lr_schedule not in ("none", "cosine"):
            raise ValueError(
                "lr_schedule must be 'none' or 'cosine'"
            )

        self.base_lr = None
        self._ema_shadow = None
        self._ema_steps = 0

        # --------------------------------------------------------
        # Architecture parameters
        # --------------------------------------------------------

        self.nFilt_FirstLayer = 25

        self.nFiltLaterLayer = [
            25,
            50,
            100,
            200,
        ]

        self.dropoutP = 0.5

        # Temporal kernel width (flag: kernel_width).
        #   "auto" -> T // 20  (your original; = 10 at T=200)
        #   int    -> fixed    (the paper uses 10)
        kw = self._cfg("kernel_width", "auto")

        if isinstance(kw, str):

            if kw != "auto":
                raise ValueError(
                    "kernel_width must be an int or 'auto'"
                )

            kw = nTime // 20

        self.kernel_width = int(kw)

        if self.kernel_width < 2:
            raise ValueError(
                f"kernel_width={self.kernel_width} is too small "
                f"(T={nTime}); set kernel_width explicitly."
            )

        # NEW (flag: temporal_pools): pool sizes for
        # [GSL, block1, block2, block3]. Default [3,3,3,3] reproduces
        # the paper (T // 81 output steps).
        pools = self._cfg("temporal_pools", [3, 3, 3, 3])
        pools = [int(p) for p in pools]

        if len(pools) != 4 or any(p < 1 for p in pools):
            raise ValueError(
                "temporal_pools must be 4 positive ints "
                "(GSL, block1, block2, block3)"
            )

        self.temporal_pools = pools

        # Kept for backwards compatibility.
        self.pool_width = pools[0]

        # Output length after all pooling layers (floor at each step,
        # exactly like MaxPool2d/AvgPool2d with stride == kernel).
        final_len = nTime

        for p in pools:
            final_len = final_len // p

        if final_len < 1:
            raise ValueError(
                f"temporal_pools {pools} reduce T={nTime} to zero "
                "steps; use smaller pools."
            )

        self.final_len = final_len

        print(
            f"EEG2GAIT: temporal_pools={pools} -> "
            f"{final_len} temporal steps before attention"
        )

        # ========================================================
        # 1. Local Temporal Learner
        # ========================================================

        self.firstLayer = nn.Sequential(

            nn.ZeroPad2d(
                (
                    self.kernel_width // 2 - 1,
                    self.kernel_width // 2,
                    0,
                    0,
                )
            ),

            Conv2dWithConstraint(
                1,
                self.nFilt_FirstLayer,
                (
                    1,
                    self.kernel_width,
                ),
                padding=0,
                max_norm=2,
            ),
        )

        # ========================================================
        # 2. Anatomical Graph
        # ========================================================

        if not hasattr(
            self.config,
            "adj_matrix",
        ):
            raise ValueError(
                "config.eegnet.adj_matrix is required for "
                "GaitGraph_tmp."
            )

        adjacency_matrix = torch.as_tensor(
            self.config.adj_matrix,
            dtype=torch.float32,
        )

        if adjacency_matrix.shape != (
            nChan,
            nChan,
        ):
            raise ValueError(
                "Adjacency matrix shape mismatch: "
                f"expected {(nChan, nChan)}, "
                f"got {tuple(adjacency_matrix.shape)}"
            )

        # Store original anatomical graph.
        #
        # This is NOT trainable. NEW: it is now used by the optional
        # prior regulariser (flag: adj_prior_weight).
        #
        self.register_buffer(
            "adjacency_prior",
            adjacency_matrix.clone(),
        )

        # --------------------------------------------------------
        # Initialize learnable adjacency from anatomical graph
        # --------------------------------------------------------

        # Flag adj_init:
        #   "raw"      -> initialise with the raw anatomical adjacency;
        #                 symmetrisation + self-loop are applied once,
        #                 in get_adj() (the paper's A_tilde).
        #   "original" -> your first version: ReLU(A + A^T) + I here,
        #                 and get_adj() applies it again (~4A + 3I).
        #
        adj_init = str(self._cfg("adj_init", "original"))

        if adj_init == "raw":

            init_adj_matrix = adjacency_matrix.clone()

        elif adj_init == "original":

            init_adj_matrix = F.relu(
                adjacency_matrix
                + adjacency_matrix.transpose(0, 1)
            )

            init_adj_matrix = init_adj_matrix + torch.eye(
                nChan,
                dtype=init_adj_matrix.dtype,
            )

        else:

            raise ValueError(
                "adj_init must be 'raw' or 'original'"
            )

        # Reference for the optional prior penalty. Not persistent, so
        # checkpoints keep the same keys as before.
        self.register_buffer(
            "adjs_init",
            init_adj_matrix.clone(),
            persistent=False,
        )

        # Two hierarchical GCN branches, each with its own learnable
        # adjacency:
        #
        # Branch 1 -> depth 1
        # Branch 2 -> depth 2
        #
        self.adjs = nn.Parameter(
            init_adj_matrix.clone().unsqueeze(0).repeat(
                2,
                1,
                1,
            ),
            requires_grad=True,
        )

        print(
            f"Initializing learnable adjacency "
            f"from anatomical EEG graph (adj_init='{adj_init}')."
        )

        # ========================================================
        # 3. Hierarchical Graph Convolutional Pyramid
        # ========================================================

        gcn_last_act = str(self._cfg("gcn_last_act", "relu"))
        gcn_norm = bool(self._cfg("gcn_norm", False))
        gcn_residual = bool(self._cfg("gcn_residual", False))

        self.GE1 = self._HierarchicalGCN(
            in_features=self.nFilt_FirstLayer,
            hidden_features=self.nFilt_FirstLayer,
            out_features=self.nFilt_FirstLayer,
            depth=1,
            last_act=gcn_last_act,
            use_norm=gcn_norm,
            residual=gcn_residual,
        )

        self.GE2 = self._HierarchicalGCN(
            in_features=self.nFilt_FirstLayer,
            hidden_features=self.nFilt_FirstLayer,
            out_features=self.nFilt_FirstLayer,
            depth=2,
            last_act=gcn_last_act,
            use_norm=gcn_norm,
            residual=gcn_residual,
        )

        # ========================================================
        # 4. Global Spatial Learner
        # ========================================================

        # Flags: gsl_depthwise, gsl_dropout, gsl_pool.
        #
        # The paper describes a depthwise (grouped) conv, dropout 0.5
        # and average pooling; your original used a full conv, no
        # dropout and max pooling.
        #
        # NOTE: gsl_depthwise assumes Conv2dWithConstraint forwards
        # **kwargs (e.g. groups) to nn.Conv2d. `groups` is only passed
        # when the flag is on.
        #
        gsl_depthwise = bool(self._cfg("gsl_depthwise", False))
        gsl_dropout = bool(self._cfg("gsl_dropout", False))
        gsl_pool = str(self._cfg("gsl_pool", "max"))

        if gsl_pool not in ("max", "avg"):
            raise ValueError("gsl_pool must be 'max' or 'avg'")

        conv_kwargs = {}

        if gsl_depthwise:
            conv_kwargs["groups"] = self.nFilt_FirstLayer

        gsl_layers = [

            Conv2dWithConstraint(
                self.nFilt_FirstLayer,
                self.nFilt_FirstLayer,
                (
                    nChan,
                    1,
                ),
                padding=0,
                bias=False,
                max_norm=2,
                **conv_kwargs,
            ),

            nn.BatchNorm2d(
                self.nFilt_FirstLayer
            ),

            nn.ELU(),
        ]

        if gsl_dropout:

            gsl_layers.append(
                nn.Dropout(
                    p=self.dropoutP
                )
            )

        pool_cls = (
            nn.MaxPool2d
            if gsl_pool == "max"
            else nn.AvgPool2d
        )

        gsl_layers.append(
            pool_cls(
                (
                    1,
                    pools[0],
                ),
                stride=(
                    1,
                    pools[0],
                ),
            )
        )

        self.spatialLayer = nn.Sequential(
            *gsl_layers
        )

        # ========================================================
        # 5. Feature Fusion Layers
        # ========================================================

        self.middleLayers = nn.Sequential(

            self.convBlock(
                25,
                50,
                self.dropoutP,
                self.kernel_width,
                pools[1],
            ),

            self.convBlock(
                50,
                100,
                self.dropoutP,
                self.kernel_width,
                pools[2],
            ),
        )

        # 100 -> 200

        self.firstGlobalLayer = self.convBlock(
            100,
            200,
            self.dropoutP,
            self.kernel_width,
            pools[3],
        )

        # ========================================================
        # 6. Global Temporal Learner
        # ========================================================

        self.attn_variant = str(self._cfg("attn_variant", "paper"))
        attn_heads = int(self._cfg("attn_heads", 4))

        if self.attn_variant == "paper":

            self.temporalGlobalLayer = nn.MultiheadAttention(
                embed_dim=200,
                num_heads=attn_heads,
                batch_first=False,
            )

        elif self.attn_variant == "enhanced":

            self.temporalGlobalLayer = self._TemporalAttentionBlock(
                d_model=200,
                num_heads=attn_heads,
                dropout=float(self._cfg("attn_dropout", 0.1)),
                ff_mult=int(self._cfg("attn_ff_mult", 2)),
                max_len=self.final_len,
                use_pos=bool(self._cfg("attn_pos_enc", True)),
            )

        else:

            raise ValueError(
                "attn_variant must be 'paper' or 'enhanced'"
            )

        # ========================================================
        # 7. Output Layer
        # ========================================================

        # Concatenated length = 2 * final_len. For the default
        # pools this equals the paper's (nTime // 81) * 2.
        self.temporalLastLayer = Conv2dWithConstraint(
            200,
            nClass,
            (
                1,
                self.final_len * 2,
            ),
            padding=0,
            max_norm=2,
        )

        # ========================================================
        # Dropout
        # ========================================================

        self.tmpdropout = nn.Dropout(
            p=self.dropoutP
        )

        # ========================================================
        # Target statistics (used when target_norm=True)
        # ========================================================

        self.register_buffer(
            "y_mean",
            torch.zeros(nClass),
        )

        self.register_buffer(
            "y_std",
            torch.ones(nClass),
        )

        self.register_buffer(
            "y_stats_set",
            torch.tensor(False),
        )

        # ========================================================
        # Loss
        # ========================================================

        if self.config.loss_name == "mse":

            self.loss_fn = nn.MSELoss()

        elif self.config.loss_name == "freq_mse":

            self.loss_fn = FreDFLoss()

        elif self.config.loss_name == "reward_mse":

            self.loss_fn = MSEWithEncouragingLoss()

        elif self.config.loss_name == "freq_reward_mse":

            self.loss_fn = FreDFLossWithEncouraging()

        else:

            print(
                f"WARNING: Unknown loss "
                f"'{self.config.loss_name}'. "
                f"Using MSELoss."
            )

            self.loss_fn = nn.MSELoss()

    # ============================================================
    # Convolution block
    # ============================================================

    def convBlock(
        self,
        inF,
        outF,
        dropoutP,
        kernelSize,
        poolSize,
    ):

        return nn.Sequential(

            nn.Dropout(
                p=dropoutP
            ),

            nn.ZeroPad2d(
                (
                    kernelSize // 2 - 1,
                    kernelSize // 2,
                    0,
                    0,
                )
            ),

            Conv2dWithConstraint(
                inF,
                outF,
                (
                    1,
                    kernelSize,
                ),
                bias=False,
                max_norm=2,
            ),

            nn.BatchNorm2d(
                outF
            ),

            nn.ELU(),

            nn.MaxPool2d(
                (
                    1,
                    poolSize,
                ),
                stride=(
                    1,
                    poolSize,
                ),
            ),
        )

    # ============================================================
    # Graph normalization
    # ============================================================

    def get_adj(
        self,
        self_loop=True,
    ):
        """
        Construct dynamically learned normalized adjacency.

        Step 1:
            A = ReLU(A + A^T)

        Step 2:
            A = A + I

        Step 3:
            D_ii = sum_j A_ij

        Step 4:
            A_hat = D^(-1/2) A D^(-1/2)

        Returns:
            (2, C, C)
        """

        # --------------------------------------------------------
        # Symmetrize
        # --------------------------------------------------------

        adj = F.relu(
            self.adjs
            + self.adjs.transpose(
                1,
                2,
            )
        )

        # --------------------------------------------------------
        # Self loops
        # --------------------------------------------------------

        if self_loop:

            eye = torch.eye(
                self.nChan,
                dtype=adj.dtype,
                device=adj.device,
            )

            adj = (
                adj
                + eye.unsqueeze(0)
            )

        # --------------------------------------------------------
        # Degree
        # --------------------------------------------------------

        degree = adj.sum(
            dim=-1
        )

        # Prevent divide-by-zero

        degree = torch.clamp(
            degree,
            min=1e-8,
        )

        # --------------------------------------------------------
        # D^(-1/2)
        # --------------------------------------------------------

        degree_inv_sqrt = degree.pow(
            -0.5
        )

        # --------------------------------------------------------
        # D^(-1/2) A D^(-1/2)
        # --------------------------------------------------------

        adj = (
            degree_inv_sqrt.unsqueeze(-1)
            * adj
            * degree_inv_sqrt.unsqueeze(-2)
        )

        return adj

    # ============================================================
    # NEW: target standardisation
    # ============================================================

    @torch.no_grad()
    def set_target_stats(
        self,
        Y,
        eps=1e-6,
    ):
        """
        Store per-joint mean/std computed from TRAINING targets.

        Y: array/tensor of shape (N, num_chan_kin).
        Call once before training when target_norm=True. The stats
        are buffers, so they are saved in / restored from checkpoints.
        """

        Y = torch.as_tensor(
            Y,
            dtype=torch.float32,
        )

        if Y.ndim != 2 or Y.shape[1] != self.nClass:
            raise ValueError(
                f"Expected targets of shape (N, {self.nClass}), "
                f"got {tuple(Y.shape)}"
            )

        mean = Y.mean(dim=0)
        std = Y.std(dim=0).clamp_min(eps)

        self.y_mean.copy_(mean.to(self.y_mean.device))
        self.y_std.copy_(std.to(self.y_std.device))
        self.y_stats_set.fill_(True)

    def _check_target_stats(self):

        if not bool(self.y_stats_set):

            raise RuntimeError(
                "target_norm=True but target statistics are not set. "
                "Call model.fit_target_stats(train_loader) or "
                "model.set_target_stats(train_Y) before training/"
                "evaluating (or load a checkpoint that contains "
                "them)."
            )

    @torch.no_grad()
    def fit_target_stats(
        self,
        loader,
        max_batches=None,
    ):
        """
        Compute the target statistics by iterating a TRAINING
        dataloader once (batches must be dicts with a "Y" entry of
        shape (B, num_chan_kin)):

            model.fit_target_stats(train_loader)

        max_batches: optionally stop early (use a shuffled loader if
        you do, so the subset is representative).
        """

        ys = []

        for i, batch in enumerate(loader):

            if max_batches is not None and i >= max_batches:
                break

            ys.append(
                torch.as_tensor(
                    batch["Y"],
                    dtype=torch.float32,
                ).detach().cpu()
            )

        if not ys:
            raise ValueError(
                "fit_target_stats: the loader yielded no batches."
            )

        self.set_target_stats(
            torch.cat(ys, dim=0)
        )

    def _normalize_targets(self, Y):

        if not self.target_norm:
            return Y

        # Fallback so training can start without an explicit call:
        # estimate the statistics from the FIRST TRAINING BATCH only.
        # This is a rough estimate (one batch, and it is skipped in
        # eval mode). Prefer model.fit_target_stats(train_loader) or
        # model.set_target_stats(train_Y) before training. Disable the
        # fallback with config.eegnet.target_stats_auto = False.
        if (
            not bool(self.y_stats_set)
            and self.training
            and bool(self._cfg("target_stats_auto", True))
        ):

            print(
                "WARNING: target statistics were not set; estimating "
                "them from the first training batch. Call "
                "model.fit_target_stats(train_loader) before training "
                "for accurate statistics."
            )

            self.set_target_stats(Y.detach().float().cpu())

        self._check_target_stats()

        return (
            Y - self.y_mean.to(Y.dtype)
        ) / self.y_std.to(Y.dtype)

    def _denormalize_predictions(self, x):

        if not self.target_norm:
            return x

        self._check_target_stats()

        return (
            x * self.y_std.to(x.dtype)
            + self.y_mean.to(x.dtype)
        )

    # ============================================================
    # NEW: extra loss terms
    # ============================================================

    def _pearson_loss(
        self,
        pred,
        target,
        eps=1e-8,
    ):
        """
        1 - mean_j Pearson r_j, with r_j computed over the batch for
        each output joint j. Needs batch size >= 2.
        """

        if pred.shape[0] < 2:
            return pred.new_zeros(())

        pred = pred.float()
        target = target.float()

        pc = pred - pred.mean(dim=0, keepdim=True)
        tc = target - target.mean(dim=0, keepdim=True)

        num = (pc * tc).sum(dim=0)

        den = torch.sqrt(
            (pc ** 2).sum(dim=0)
            * (tc ** 2).sum(dim=0)
        ) + eps

        return 1.0 - (num / den).mean()

    def _adj_prior_penalty(self):
        """Penalty keeping the learned graph close to the prior."""

        deviation = (
            self.adjs
            - self.adjs_init.unsqueeze(0)
        )

        if self.adj_prior_type == "l1":
            return deviation.abs().mean()

        return deviation.pow(2).mean()

    # ============================================================
    # NEW: augmentation (training only)
    # ============================================================

    def _augment(self, x):
        """
        x: (B, 1, C, T). Applies whichever augmentations are enabled.

        Note: time shifts / noise / gain jitter the input only; the
        label is left unchanged, so keep the shifts small relative to
        the window.
        """

        B, _, C, T = x.shape

        # Random time shift (edge-replicated, per sample)
        if self.aug_time_shift > 0:

            shifts = torch.randint(
                -self.aug_time_shift,
                self.aug_time_shift + 1,
                (B,),
                device=x.device,
            )

            idx = (
                torch.arange(T, device=x.device).unsqueeze(0)
                + shifts.unsqueeze(1)
            ).clamp(0, T - 1)

            idx = idx.view(B, 1, 1, T).expand(B, 1, C, T)

            x = torch.gather(x, 3, idx)

        # Random gain
        if self.aug_amp_scale > 0:

            gain = 1.0 + (
                torch.rand(
                    B,
                    1,
                    1,
                    1,
                    device=x.device,
                    dtype=x.dtype,
                )
                * 2.0
                - 1.0
            ) * self.aug_amp_scale

            x = x * gain

        # Gaussian noise scaled by each channel's own std
        if self.aug_noise_std > 0:

            sigma = x.std(
                dim=-1,
                keepdim=True,
            )

            x = x + (
                torch.randn_like(x)
                * sigma
                * self.aug_noise_std
            )

        # Random channel dropout (zeroes whole channels)
        if self.aug_channel_drop > 0:

            keep = (
                torch.rand(
                    B,
                    1,
                    C,
                    1,
                    device=x.device,
                )
                >= self.aug_channel_drop
            ).to(x.dtype)

            x = x * keep

        return x

    # ============================================================
    # Forward
    # ============================================================

    def forward(
        self,
        batch,
        metrics=None,
    ):

        if metrics is None:
            metrics = []

        # ========================================================
        # Input
        # ========================================================

        x = batch["EEG"]
        Y = batch["Y"]

        # Expected:
        #
        # (B, 1, C, T)
        #

        if x.ndim != 4:

            raise ValueError(
                "Expected EEG input with shape "
                "(B, 1, C, T), "
                f"got {tuple(x.shape)}"
            )

        B = x.shape[0]

        # NEW (flags: aug_*): training-time augmentation
        if self.training:
            x = self._augment(x)

        # ========================================================
        # 1. Local Temporal Learner
        # ========================================================

        x = self.firstLayer(x)

        # Expected:
        #
        # (B, 25, C, T)
        #

        # ========================================================
        # 2. Prepare graph input
        # ========================================================

        # B, F, C, T
        #
        # ->
        #
        # B, T, C, F

        x_gcn = x.permute(
            0,
            3,
            2,
            1,
        )

        B, T, C, F_dim = x_gcn.shape

        if C != self.nChan:

            raise ValueError(
                f"EEG has {C} channels, "
                f"but model expects {self.nChan}."
            )

        # --------------------------------------------------------
        # Treat every time point as an independent graph
        # --------------------------------------------------------

        x_reshaped = rearrange(
            x_gcn,
            "b t c f -> (b t) c f",
        )

        # Shape:
        #
        # (B*T, C, 25)
        #

        # ========================================================
        # 3. Dynamic normalized adjacency
        # ========================================================

        adjs = self.get_adj(
            self_loop=True
        )

        # ========================================================
        # 4. Hierarchical GCN Pyramid
        # ========================================================

        # --------------------------------------------------------
        # Branch 1
        # --------------------------------------------------------

        x_gcn1 = self.GE1(
            x_reshaped,
            adjs[0],
        )

        # (B*T, C, 25)

        # --------------------------------------------------------
        # Branch 2
        # --------------------------------------------------------

        x_gcn2 = self.GE2(
            x_reshaped,
            adjs[1],
        )

        # (B*T, C, 25)

        # ========================================================
        # Restore dimensions
        # ========================================================

        x_gcn1 = rearrange(
            x_gcn1,
            "(b t) c f -> b f c t",
            b=B,
            t=T,
        )

        x_gcn2 = rearrange(
            x_gcn2,
            "(b t) c f -> b f c t",
            b=B,
            t=T,
        )

        # ========================================================
        # Hierarchical graph fusion
        # ========================================================

        # NOTE: the paper only says the two encoder outputs are
        # "fused" (Fig. 1 caption says "accumulated"). Averaging is
        # kept; switch to a plain sum to follow the caption literally.
        x_graph = (
            x_gcn1
            + x_gcn2
        ) / 2.0

        # Residual connection with local temporal features

        x = x + x_graph

        # Shape:
        #
        # (B, 25, C, T)
        #

        # ========================================================
        # 5. Global Spatial Learner
        # ========================================================

        x = self.spatialLayer(x)

        # Expected:
        #
        # (B, 25, 1, T // pools[0])
        #

        # ========================================================
        # 6. Feature Fusion
        # ========================================================

        x = self.middleLayers(x)

        # (B, 100, 1, T // (pools[0] * pools[1] * pools[2]))

        x = self.firstGlobalLayer(x)

        # (B, 200, 1, final_len)

        # ========================================================
        # Remove spatial dimension
        # ========================================================

        x = torch.squeeze(
            x,
            dim=2,
        )

        # (B, 200, final_len)

        # ========================================================
        # 7. Global Temporal Learner
        # ========================================================

        x_original = self.tmpdropout(
            x
        )

        if self.attn_variant == "paper":

            # MHA expects:
            #
            # (sequence, batch, embedding)

            x_attention_input = x_original.permute(
                2,
                0,
                1,
            )

            # (final_len, B, 200)

            x_tempo, _ = self.temporalGlobalLayer(
                x_attention_input,
                x_attention_input,
                x_attention_input,
            )

            # Restore:
            #
            # (B, 200, final_len)

            x_tempo = x_tempo.permute(
                1,
                2,
                0,
            )

            # ====================================================
            # Temporal residual connection
            # ====================================================

            x_tempo = (
                x_tempo
                + x_original
            )

        else:

            # NEW (flag: attn_variant="enhanced"): pre-norm block
            # with positional encoding + FFN; residuals are inside.
            x_tempo = self.temporalGlobalLayer(
                x_original
            )

        # ========================================================
        # 8. Concatenate original + temporal features
        # ========================================================

        x = torch.cat(
            (
                x_original,
                x_tempo,
            ),
            dim=2,
        )

        # (B, 200, 2 * final_len)

        # Add spatial dimension

        x = x.unsqueeze(
            dim=2
        )

        # (B, 200, 1, 2 * final_len)

        # ========================================================
        # 9. Output layer
        # ========================================================

        x = self.temporalLastLayer(
            x
        )

        # (B, nClass, 1, 1)

        x = torch.squeeze(
            x,
            dim=3,
        )

        x = torch.squeeze(
            x,
            dim=2,
        )

        # (B, nClass)

        # ========================================================
        # Loss
        # ========================================================

        # NEW (flag: target_norm): the network predicts / is trained
        # in z-scored space; predictions are mapped back to the
        # original units before being returned.
        Y_t = self._normalize_targets(Y)

        loss_main = self.loss_fn(
            x,
            Y_t,
        )

        x_pred = self._denormalize_predictions(
            x
        ).to(torch.float32)

        # ========================================================
        # Validation / inference
        # ========================================================

        # The validation loss is the base loss only, so it stays
        # comparable across runs with different extra loss terms.

        if not self.training:

            return (
                loss_main.item(),
                x_pred,
            )

        # ========================================================
        # NEW: extra training-only loss terms
        # ========================================================

        loss = loss_main

        loss_corr = None
        loss_prior = None

        if self.corr_loss_weight > 0:

            loss_corr = self._pearson_loss(
                x,
                Y_t,
            )

            loss = loss + self.corr_loss_weight * loss_corr

        if self.adj_prior_weight > 0:

            loss_prior = self._adj_prior_penalty()

            loss = loss + self.adj_prior_weight * loss_prior

        # ========================================================
        # Metrics
        # ========================================================

        results = {}

        if "r2" in metrics:

            with torch.amp.autocast(
                "cuda",
                enabled=False,
            ):

                results = get_metrics(
                    x_pred
                    .detach()
                    .cpu()
                    .numpy(),

                    Y.cpu().numpy(),

                    metrics,

                    is_binary=False,
                )

        # ========================================================
        # Logging
        # ========================================================

        log = {}

        split = (
            "train"
            if self.training
            else "val"
        )

        log[
            f"{split}/loss_total"
        ] = loss.item()

        log[
            f"{split}/loss_main"
        ] = loss_main.item()

        if loss_corr is not None:

            log[
                f"{split}/loss_corr"
            ] = loss_corr.item()

        if loss_prior is not None:

            log[
                f"{split}/loss_adj_prior"
            ] = loss_prior.item()

        for key, value in results.items():

            log[
                f"{split}/{key}"
            ] = value

        return loss, log

    # ============================================================
    # Optimizer
    # ============================================================

    def configure_optimizers(
        self,
        weight_decay,
        learning_rate,
        betas,
        device_type,
    ):

        # NEW: remembered for get_lr() / apply_lr().
        self.base_lr = learning_rate

        # --------------------------------------------------------
        # Trainable parameters
        # --------------------------------------------------------

        param_dict = {
            pn: p
            for pn, p in self.named_parameters()
            if p.requires_grad
        }

        # --------------------------------------------------------
        # Weight decay
        # --------------------------------------------------------

        # CHANGED: the learnable adjacency (3-D) is excluded from
        # weight decay; decay would shrink it toward zero and, with
        # the fixed +I in get_adj(), collapse the graph to self-loops.
        #
        # For a paper-faithful setup (plain Adam, no decay) also pass
        # weight_decay=0.0 from the training config.

        no_decay_names = (
            {"adjs"}
            if self.adj_no_decay
            else set()
        )

        decay_params = [
            p
            for pn, p in param_dict.items()
            if p.dim() >= 2 and pn not in no_decay_names
        ]

        nodecay_params = [
            p
            for pn, p in param_dict.items()
            if p.dim() < 2 or pn in no_decay_names
        ]

        optim_groups = [
            {
                "params": decay_params,
                "weight_decay": weight_decay,
            },
            {
                "params": nodecay_params,
                "weight_decay": 0.0,
            },
        ]

        num_decay_params = sum(
            p.numel()
            for p in decay_params
        )

        num_nodecay_params = sum(
            p.numel()
            for p in nodecay_params
        )

        print(
            "num decayed parameter tensors: "
            f"{len(decay_params)}, "
            f"with {num_decay_params:,} parameters"
        )

        print(
            "num non-decayed parameter tensors: "
            f"{len(nodecay_params)}, "
            f"with {num_nodecay_params:,} parameters"
        )

        # --------------------------------------------------------
        # AdamW
        # --------------------------------------------------------

        fused_available = (
            "fused"
            in inspect.signature(
                torch.optim.AdamW
            ).parameters
        )

        use_fused = (
            fused_available
            and device_type == "cuda"
        )

        extra_args = (
            {"fused": True}
            if use_fused
            else {}
        )

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args,
        )

        print(
            f"using fused AdamW: {use_fused}"
        )

        return optimizer

    # ============================================================
    # NEW: learning-rate schedule
    # ============================================================

    def get_lr(self, it):
        """
        Learning rate for iteration `it` (flag: lr_schedule).

        "none"   -> constant base LR
        "cosine" -> linear warmup, then cosine decay to
                    base_lr * min_lr_ratio at lr_decay_iters
        """

        if self.base_lr is None:
            raise RuntimeError(
                "Call configure_optimizers() before get_lr()."
            )

        if self.lr_schedule == "none":
            return self.base_lr

        min_lr = self.base_lr * self.min_lr_ratio

        if it < self.warmup_iters:
            return self.base_lr * (it + 1) / (self.warmup_iters + 1)

        # lr_decay_iters is unknown by default (it depends on how long
        # you train). Until you set it, the schedule is warmup followed
        # by a constant LR.
        if self.lr_decay_iters <= 0:

            if not getattr(self, "_warned_no_decay", False):
                print(
                    "WARNING: lr_schedule='cosine' but lr_decay_iters "
                    "is not set; using warmup + constant LR. Set "
                    "config.eegnet.lr_decay_iters to enable decay."
                )
                self._warned_no_decay = True

            return self.base_lr

        if it >= self.lr_decay_iters:
            return min_lr

        span = max(1, self.lr_decay_iters - self.warmup_iters)

        ratio = (it - self.warmup_iters) / span

        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))

        return min_lr + coeff * (self.base_lr - min_lr)

    def apply_lr(self, optimizer, it):
        """Set the scheduled LR on every param group and return it."""

        lr = self.get_lr(it)

        for group in optimizer.param_groups:
            group["lr"] = lr

        return lr

    # ============================================================
    # NEW: gradient clipping
    # ============================================================

    def clip_gradients(self):
        """
        Clip the global grad norm (flag: grad_clip; 0 = off).

        With a GradScaler, call scaler.unscale_(optimizer) first.
        Returns the pre-clip norm, or None when disabled.
        """

        if self.grad_clip <= 0:
            return None

        return torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            self.grad_clip,
        )

    # ============================================================
    # NEW: exponential moving average of the weights
    # ============================================================

    def _ema_tensors(self):
        """Parameters plus BatchNorm running statistics."""

        for name, p in self.named_parameters():
            yield name, p

        for name, b in self.named_buffers():

            if name.endswith("running_mean") or name.endswith(
                "running_var"
            ):
                yield name, b

    @torch.no_grad()
    def ema_update(self):
        """
        Call once after every optimizer step (flag: ema_decay > 0).

        Uses a warm-up ramp d = min(ema_decay, (1+n)/(10+n)) so the
        average is not dominated by the random initial weights.
        """

        if self.ema_decay <= 0:
            return

        if self._ema_shadow is None:

            self._ema_shadow = {
                name: t.detach().clone()
                for name, t in self._ema_tensors()
            }

            self._ema_steps = 0

            return

        self._ema_steps += 1

        d = min(
            self.ema_decay,
            (1.0 + self._ema_steps) / (10.0 + self._ema_steps),
        )

        for name, t in self._ema_tensors():

            self._ema_shadow[name].mul_(d).add_(
                t.detach(),
                alpha=1.0 - d,
            )

    @contextlib.contextmanager
    def ema_scope(self):
        """
        Temporarily swap in the EMA weights, e.g. for validation:

            model.eval()
            with model.ema_scope():
                ... run validation / test ...

        A no-op if EMA is disabled or has not started yet.
        """

        if self._ema_shadow is None:
            yield
            return

        backup = {
            name: t.detach().clone()
            for name, t in self._ema_tensors()
        }

        try:

            with torch.no_grad():

                for name, t in self._ema_tensors():
                    t.copy_(self._ema_shadow[name])

            yield

        finally:

            with torch.no_grad():

                for name, t in self._ema_tensors():
                    t.copy_(backup[name])

    def ema_state_dict(self):
        """EMA shadow weights (for checkpointing), or None."""

        return self._ema_shadow

    def load_ema_state_dict(self, state, steps=0):
        """Restore EMA shadow weights saved via ema_state_dict()."""

        self._ema_shadow = (
            None
            if state is None
            else {k: v.clone() for k, v in state.items()}
        )

        self._ema_steps = int(steps)