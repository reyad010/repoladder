# Description: This file contains the implementation of the Layers.

import sys
import time
from torch import Tensor
import torch.nn
from types import MethodType
sys.path.append('../utils/')
from utils.pytorch_utils import *
from functools import partial
from typing import Dict, Optional, Tuple
import math
from utils.flops_computation import conv2d_flops, linear_flops, reduced_conv_out
from utils.mha_debug_tools import instrument_mha
import torch.nn.functional as F
import copy
# from transformers.models.t5.modeling_t5 import T5Block, T5Config
T5Block, T5Config = None, None
lora_r = 16
def forward_no_fastpath(
    self,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Forced no-fastpath forward for MultiheadAttention.
    Always uses F.multi_head_attention_forward.
    """

    # Handle shape: batch_first
    is_batched = query.dim() == 3
    if self.batch_first and is_batched:
        query, key, value = (x.transpose(1, 0) for x in (query, key, value))

    # Canonicalize masks
    key_padding_mask = F._canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=F._none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype,
    )
    attn_mask = F._canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=None,
        other_name="",
        target_type=query.dtype,
        check_other=False,
    )

    # Disable all fastpath logic intentionally
    if not self._qkv_same_embed_dim:
        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj_weight,
            k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
    else:
        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

    # Restore batch_first layout if needed
    if self.batch_first and is_batched:
        attn_output = attn_output.transpose(1, 0)

    return attn_output, attn_output_weights

def _expand_mask_4d(mask_2d: Optional[torch.Tensor], dtype: torch.dtype, tgt_len: int) -> Optional[torch.Tensor]:
    """
    (B, S) -> (B, 1, tgt_len, S) additive mask (0 keep, -inf mask), as expected by T5Block.
    """
    if mask_2d is None:
        return None
    if mask_2d.dim() != 2:
        raise ValueError(f"attention_mask must be shape (B,S), got {tuple(mask_2d.shape)}")
    bsz, src_len = mask_2d.shape
    add_mask = (1.0 - mask_2d.to(dtype)) * torch.finfo(dtype).min
    return add_mask.view(bsz, 1, 1, src_len).expand(bsz, 1, tgt_len, src_len)


def _mk_reduced_cfg(
    embed_dim: int,
    num_heads: int,
    reduction_factor: int,
    block_type: str,
    base_cfg: Optional[T5Config] = None,
):
    """
    Create a T5Config for a single reduced-width T5Block (Option A):
    - d_model_inner = max(1, embed_dim // reduction_factor)
    - keep num_heads unchanged
    - keep d_kv unchanged (from base_cfg if provided; else fallback from embed_dim/num_heads)
    - scale d_ff proportionally to d_model
    Other hyper-params come from base_cfg when provided, else T5-base-like defaults.
    """
    rf = max(int(reduction_factor), 1)
    d_model_outer = int(embed_dim)
    d_model_inner = max(1, d_model_outer // rf)
    is_dec = (block_type == "decoder")

    if base_cfg is None:
        # Fallbacks when no base config is provided
        # Keep heads as given; choose a reasonable d_kv fallback
        d_kv_fallback = max(1, d_model_outer // int(num_heads))
        cfg_dict = dict(
            vocab_size=32128,
            d_model=d_model_inner,
            d_kv=d_kv_fallback,
            d_ff=max(1, 4 * d_model_inner),
            num_layers=1,
            num_decoder_layers=1,
            num_heads=int(num_heads),
            relative_attention_num_buckets=32,
            dropout_rate=0.1,
            layer_norm_epsilon=1e-6,
            feed_forward_proj="relu",
            is_decoder=is_dec,
            use_cache=False,
        )
    else:
        cfg_dict = copy.deepcopy(base_cfg.to_dict())
        # Keep heads & d_kv from base config; scale d_model/d_ff only
        d_kv_keep = getattr(base_cfg, "d_kv", max(1, d_model_outer // int(num_heads)))
        d_ff_scaled = max(1, int(round(base_cfg.d_ff * (d_model_inner / max(1, base_cfg.d_model)))))

        cfg_dict.update(
            dict(
                d_model=d_model_inner,
                d_kv=d_kv_keep,
                d_ff=d_ff_scaled,
                num_heads=base_cfg.num_heads,   # unchanged
                is_decoder=is_dec,
                num_layers=1,
                num_decoder_layers=1,
                use_cache=False,
            )
        )

    return T5Config(**cfg_dict)

class ReducedGroupingLayer(nn.Module):
    pass

class EmptyLayer(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, padding=0,
                 use_bn=False, act_func=None, dropout_rate=0.0, type='connection'):
        super(EmptyLayer, self).__init__()

        self.input_shape = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.type = type

    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = getattr(x, 'shape', None)
        if self.type == 'connection':
            return None
        res = x
        return res

    @property
    def module_str(self):
        return 'IdentityLayer'

    @property
    def config(self):
        return {
            'name': EmptyLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
        }

    def get_model_size(self):
        return 0

    def is_zero_layer(self):
        return False

    def get_flops(self):
        return 0, (None if self.type == 'connection' else self.input_shape)

class ReducedConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=True, has_shuffle=False,
                 use_bn=False, act_func=None, dropout_rate=0.0, type='connection', reduction_factor=1):
        super(ReducedConvLayer, self).__init__()

        self.input_shape = None
        self.type = type
        self.reduction_factor = reduction_factor

        if type == 'connection':
            self.in_channels = int(in_channels)
            self.out_channels = int(out_channels / reduction_factor)
        elif type == 'side':
            self.in_channels = int(in_channels / reduction_factor)
            self.out_channels = int(out_channels / reduction_factor)
        else:
            raise ValueError('type must be either "connection" or "side"')

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle
        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate

        """ modules """
        modules = {}
        modules['conv'] = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, dilation=dilation,
                                    groups=groups, bias=bias)
        self.add_module('conv', modules['conv'])

        if self.use_bn:
            modules['bn'] = nn.BatchNorm2d(self.out_channels)
            self.add_module('bn', modules['bn'])

        if self.groups > 1 and self.has_shuffle:
            modules['shuffle'] = ShuffleLayer(groups=self.groups)
            self.add_module('shuffle', modules['shuffle'])

        if self.act_func is not None:
            modules['activation'] = build_activation(act_func)
            self.add_module('activation', modules['activation'])

    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.shape
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                return '%dx%d_DilatedConv' % (kernel_size[0], kernel_size[1])
            else:
                return '%dx%d_Conv' % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                return '%dx%d_DilatedGroupConv' % (kernel_size[0], kernel_size[1])
            else:
                return '%dx%d_GroupConv' % (kernel_size[0], kernel_size[1])

    @property
    def config(self):
        return {
            'name': ReducedConvLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
        }

    def get_model_size(self):

        model_size = 0

        for module in self._modules.values():

            if isinstance(module, nn.Conv2d):
                model_size += module.weight.numel()
                if module.bias is not None:
                    model_size += module.bias.numel()

        return model_size

    @staticmethod
    def is_zero_layer():
        return False

    def get_flops(self):
        N, C_in, H_in, W_in = self.input_shape
        # ---------- unpack Conv2d parameters ----------
        k_h, k_w = self.kernel_size if isinstance(self.kernel_size, (tuple, list)) else (self.kernel_size,) * 2
        s_h, s_w = self.stride if isinstance(self.stride, (tuple, list)) else (self.stride,) * 2
        p_h, p_w = self.padding if isinstance(self.padding, (tuple, list)) else (self.padding,) * 2
        d_h, d_w = self.dilation if isinstance(self.dilation, (tuple, list)) else (self.dilation,) * 2

        H_out = math.floor((H_in + 2*p_h - d_h*(k_h-1) - 1)/s_h + 1)
        W_out = math.floor((W_in + 2*p_w - d_w*(k_w-1) - 1)/s_w + 1)
        per = conv2d_flops(C_in, self.out_channels, k_h, k_w, H_out, W_out, self.groups)
        return N*per, (N, self.out_channels, H_out, W_out)

class ReducedLinearLayer(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=True, has_shuffle=False,
                 use_bn=False, act_func=None, dropout_rate=0.0, type='connection', reduction_factor=1):
        super(ReducedLinearLayer, self).__init__()

        self.input_shape = None
        self.type = type
        self.reduction_factor = reduction_factor

        if type == 'connection':
            self.in_channels = int(in_channels)
            self.out_channels = int(out_channels / reduction_factor)
        elif type == 'side':
            self.in_channels = int(in_channels / reduction_factor)
            self.out_channels = int(out_channels / reduction_factor)
        elif type == 'upsample':
            self.in_channels = int(in_channels / reduction_factor)
            self.out_channels = int(out_channels)
        else: raise NotImplementedError


        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle
        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate


        """ modules """
        modules = {}
        # modules['conv'] = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
        #                             dilation=dilation, groups=groups, bias=bias)
        modules['linear'] = nn.Linear(self.in_channels, self.out_channels, bias=bias)
        self.add_module('linear', modules['linear'])

        if self.use_bn:
            modules['bn'] = nn.BatchNorm2d(self.out_channels)
            self.add_module('bn', modules['bn'])

        if self.groups > 1 and self.has_shuffle:
            modules['shuffle'] = ShuffleLayer(groups=self.groups)
            self.add_module('shuffle', modules['shuffle'])

        if self.act_func is not None:
            modules['activation'] = build_activation(act_func)
            self.add_module('activation', modules['activation'])

    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.shape
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                return '%dx%d_DilatedConv' % (kernel_size[0], kernel_size[1])
            else:
                return '%dx%d_Conv' % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                return '%dx%d_DilatedGroupConv' % (kernel_size[0], kernel_size[1])
            else:
                return '%dx%d_GroupConv' % (kernel_size[0], kernel_size[1])

    @property
    def config(self):
        return {
            'name': ReducedLinearLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
        }

    def get_model_size(self):

        model_size = 0

        for module in self._modules.values():

            if isinstance(module, nn.Linear):
                model_size += module.weight.numel()
                if module.bias is not None:
                    model_size += module.bias.numel()

        return model_size

    @staticmethod
    def is_zero_layer():
        return False

    def get_flops(self):
        input_shape = self.input_shape
        if len(input_shape) == 2:
            N, D = input_shape
            return linear_flops(D, self.out_channels, N), (N, self.out_channels)
        elif len(input_shape) == 3:
            B, S, D = input_shape
            return linear_flops(D, self.out_channels, B*S), (B, S, self.out_channels)
        else:
            raise ValueError('expected 2D or 3D input')

class ViTBlock(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        reduction_factor=1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        final_norm = True,
    ) -> None:
        super(ViTBlock, self).__init__()

        self.input_shape = None
        self.reduction_factor = reduction_factor
        dim_side = embed_dim // reduction_factor
        # --- core ViT encoder -------------------------------------------------
        self.norm1 = norm_layer(dim_side)


        self.attn = nn.MultiheadAttention(
            embed_dim=dim_side,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,  # (B,N,D)
        )
        self.attn.forward = MethodType(forward_no_fastpath, self.attn)

        # self.attn.forward = force_no_fastpath_forward.__get__(self.attn, type(self.attn))
        # unpatch = instrument_mha(self.attn)

        self.drop = nn.Dropout(dropout)

        self.norm2 = norm_layer(dim_side)
        hidden_dim = int(dim_side * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim_side, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim_side),
            nn.Dropout(dropout),
        )


        if final_norm:
            # using or not using relies on if cur layer is the last layer in side net
            self.side_norm = norm_layer(dim_side)

        # -------------------------------------------------------------------------

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        if self.input_shape is None:self.input_shape = input.shape
        x = self.norm1(input)
        # torch.cuda.synchronize()
        # t0 = time.time_ns()
        x, _ = self.attn(x, x, x, need_weights=False)

        # x, _ = self.attn(self.norm1(input), self.norm1(input), self.norm1(input), need_weights=False)
        # x0, _ = self.attn(x, x, x, need_weights=False)
        # x1, _ = self.attn(x, x, x, need_weights=False)
        # print(f'{x0.reshape(-1)[-5:]}, {x0.reshape(-1)[:5]}, {x1.reshape(-1)[-5:]}, {x1.reshape(-1)[:5]}, ')
        # self.attn.forward = MethodType(forward_no_fastpath, self.attn)

        # torch.cuda.synchronize()
        # t1 = time.time_ns()

        x = self.drop(x)
        x = x + input
        # torch.cuda.synchronize()
        y = self.norm2(x)
        # torch.cuda.synchronize()
        # t2 = time.time_ns()
        y = self.mlp(y)
        # t3 = time.time_ns()
        # print(f'{input.shape}, rf: {self.reduction_factor}, {(t1 - t0) / 1e6:.2f} ms, {(t3 - t2) / 1e6:.2f} ms')
        return x + y

    # def _core_block(self, x: torch.Tensor) -> torch.Tensor:
    #     """standard ViT encoder block (LN-Attn-Drop + LN-MLP-Drop)."""
    #     # Self-attention + residual
    #     # torch.cuda.synchronize()
    #     # t0 = time.time_ns()
    #     x = self.norm1(x)
    #     y, _ = self.attn(x, x, x, need_weights=False)
    #     # torch.cuda.synchronize()
    #     # t1 = time.time_ns()
    #     x = x + self.drop(y)
    #     # torch.cuda.synchronize()
    #     # t2 = time.time_ns()
    #     # MLP + residual
    #     y = self.mlp(self.norm2(x)) # self.norm2(x)
    #     # torch.cuda.synchronize()
    #     # t3 = time.time_ns()
    #
    #     x = x + y
    #     # torch.cuda.synchronize()
    #     # t4 = time.time_ns()
    #     # print(f'vit block: {x.shape}', end='\t')
    #     # print(f'rf: {self.reduction_factor}, {(t1-t0)/1e6:.2f} ms, {(t2-t1)/1e6:.2f} ms, {(t3-t2)/1e6:.2f} ms, {(t4-t3)/1e6:.2f} ms, ')
    #     return x
    #
    # def forward(self, x_side: torch.Tensor,) -> torch.Tensor:
    #     if self.input_shape is None:
    #         self.input_shape = x_side.shape
    #     return self._core_block(x_side)

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                return '%dx%d_DilatedConv' % (kernel_size[0], kernel_size[1])
            else:
                return '%dx%d_Conv' % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                return '%dx%d_DilatedGroupConv' % (kernel_size[0], kernel_size[1])
            else:
                return '%dx%d_GroupConv' % (kernel_size[0], kernel_size[1])

    @property
    def config(self):
        return {
            'name': ViTBlock.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
        }

    def get_model_size(self):

        model_size = 0

        for module in self._modules.values():

            if isinstance(module, nn.Linear):
                model_size += module.weight.numel()
                if module.bias is not None:
                    model_size += module.bias.numel()

        return model_size

    @staticmethod
    def is_zero_layer():
        return False


    def get_flops(self):
        B, N, D = self.input_shape  # (batch, tokens, dim)
        # MH‑Attention: Q,K,V = 3·B·N·D·D + softmax negligible + proj B·N·D·D
        attn_flops = 4 * linear_flops(D, D, B*N)
        # MLP 2 linear
        hidden = self.mlp[0].out_features
        mlp_flops = linear_flops(D, hidden, B*N) + linear_flops(hidden, D, B*N)
        total = attn_flops + mlp_flops
        return total, self.input_shape

class LAConv2d(nn.Module):
    """Low‑Rank *or* Adapter replacement for Conv2d.

    * If ``layer_type`` contains ``"lora"`` the module implements **LoRA**:
        y = Δ(x)   where   Δ(x) = (B ∘ A)(x) * (α / r)

    * If ``layer_type`` contains ``"adapter"`` the module implements a **Houlsby‑style adapter**:
        y = x + Δ(x)   (residual connection is applied **only** if the channel
        dimensions match so shapes stay consistent).

    There is **no frozen base convolution** – the original Conv2d is *fully
    replaced* as per the user's request.
    """

    def __init__(
        self,
        rf: int,
        type: str,
        in_channels: int,
        out_channels: int,
        stride:int = 1,
        bias: bool = False,
        rank: int = lora_r,
        alpha: float = lora_r,
        use_bn: bool = False,
    ) -> None:
        super().__init__()

        self.input_shape = None
        self.type = type.lower()
        self.use_bn = use_bn
        self.rank = int(rank)
        self.cfg = {
            "rf": rf,
            "type": type,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "bias": bias,
            "rank": rank,
            "alpha": alpha,
            "use_bn": use_bn,
        }

        out_channels = out_channels // rf

        # Down‑ and up‑projection are always 1×1 for efficiency.
        self.down = nn.Conv2d(in_channels, self.rank, kernel_size=1, stride=stride, bias=bias)
        self.up = nn.Conv2d(self.rank, out_channels, kernel_size=1, bias=bias)

        # Activation only for adapter.
        self.act = nn.ReLU(inplace=True) if "adapter" in self.type else None

        # LoRA scaling (α / r).  If α is not provided we default to r so scaling=1.
        self.alpha = float(alpha) if alpha is not None else float(self.rank)
        self.scaling = self.alpha / float(self.rank)

        if use_bn: self.layer_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, C_in, H, W)
        if self.input_shape is None:
            self.input_shape = x.shape
        x = self.down(x)
        if self.act is not None:
            x = self.act(x)
        x = self.up(x)
        if self.use_bn:
            x = self.layer_norm(x)


        return x
    @classmethod
    def from_config(cls, cfg: Dict) -> "LAConv2d":
        return cls(
            rf = cfg["rf"],
            type=cfg["type"],
            in_channels=cfg["in_channels"],
            out_channels=cfg["out_channels"],
            stride=cfg["stride"],
            bias=cfg.get("bias", True),
            rank=cfg.get("rank", 16),
            alpha=cfg.get("alpha", 16),
        )

    def get_model_size(self):

        model_size = 0

        for module in self._modules.values():

            if isinstance(module, nn.Conv2d):
                model_size += module.weight.numel()
                if module.bias is not None:
                    model_size += module.bias.numel()

        return model_size

    @property
    def module_str(self):
        return "NoImplementation"

    @property
    def config(self):
        return self.cfg

    @staticmethod
    def is_zero_layer():
        return False

    def get_flops(self):
        N, C_in, H, W = self.input_shape
        # 1×1 convs: k=1 => out_h/out_w = H/W
        per_down = conv2d_flops(C_in, self.rank, 1, 1, H, W)
        per_up   = conv2d_flops(self.rank, self.up.out_channels, 1, 1, H, W)
        return N*(per_down+per_up), (N, self.up.out_channels, H, W)

class LALinear(nn.Module):
    """Low‑Rank *or* Adapter replacement for nn.Linear."""

    def __init__(
        self,
        rf: int,
        type: str,
        in_features: int,
        out_features: int,
        bias: bool = False,
        rank: int = lora_r,
        alpha: float = lora_r,
        use_bn: bool = False,
    ) -> None:
        super().__init__()

        self.input_shape = None
        self.type = type.lower()
        self.use_bn = use_bn
        self.rank = int(rank)
        self.cfg = {
            "rf": rf,
            "type": type,
            "in_features": in_features,
            "out_features": out_features,
            "bias": bias,
            "rank": rank,
            "alpha": alpha,
            "use_bn": use_bn,
        }

        out_features = out_features // rf

        self.down = nn.Linear(in_features, self.rank, bias=bias)
        self.up = nn.Linear(self.rank, out_features, bias=bias)

        self.act = nn.ReLU(inplace=True) if "adapter" in self.type else None
        self.alpha = float(alpha) if alpha is not None else float(self.rank)
        self.scaling = self.alpha / float(self.rank)

        if use_bn: self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (..., in_features)
        if self.input_shape is None:
            self.input_shape = x.shape
        x = self.down(x)
        if self.act is not None:
            x = self.act(x)
        x = self.up(x)
        if self.use_bn:
            x = self.layer_norm(x)

        return x

    @classmethod
    def from_config(cls, cfg: Dict) -> "LALinear":
        return cls(
            rf= cfg["rf"],
            type=cfg["type"],
            in_features=cfg["in_features"],
            out_features=cfg["out_features"],
            bias=cfg.get("bias", True),
            rank=cfg.get("rank", 16),
            alpha=cfg.get("alpha", 16),
            use_bn=cfg.get("use_bn", True),
        )

    def get_model_size(self):

        model_size = 0

        for module in self._modules.values():

            if isinstance(module, nn.Linear):
                model_size += module.weight.numel()
                if module.bias is not None:
                    model_size += module.bias.numel()

        return model_size

    @property
    def module_str(self):
        return "NoImplementation"

    @property
    def config(self):
        return self.cfg

    @staticmethod
    def is_zero_layer():
        return False

    def get_flops(self):
        input_shape = self.input_shape
        if len(input_shape) == 2:
            N, D = input_shape
            return linear_flops(D, self.rank, N) + linear_flops(self.rank, self.up.out_features, N), (
            N, self.up.out_features)
        elif len(input_shape) == 3:
            B, S, D = input_shape
            total = linear_flops(D, self.rank, B * S) + linear_flops(self.rank, self.up.out_features, B * S)
            return total, (B, S, self.up.out_features)
        else:
            raise ValueError('LALinear expects 2D/3D input')

# class ReducedT5Block(T5Block):
#     """
#     Reduced T5 block with the *same outer interface* you used previously:
#
#         layer = ReducedT5Block(embed_dim, num_heads, reduction_factor=rf, type='encoder')
#         layer = ReducedT5Block(embed_dim, num_heads, reduction_factor=rf, type='decoder')
#
#     - Inherits HF's T5Block (so its internal forward is the same).
#     - Only customizes __init__ to construct a reduced T5Config.
#     - Adds `set_memory()` so your side-decoder path can do layer.set_memory(...).
#     - Minimal forward override: if `is_decoder` and no memory is passed in this call,
#       it injects the stored memory; also expands 2D masks to 4D additive masks.
#
#     If you prefer *zero* override, remove the forward below and always pass
#     (encoder_hidden_states, encoder_attention_mask) explicitly at call site.
#     """
#     def __init__(
#         self,
#         embed_dim: int,
#         num_heads: int,
#         reduction_factor: int = 1,
#         type: str = "encoder",   # "encoder" | "decoder"
#         has_relative_attention_bias: bool = False,
#         base_cfg: Optional[T5Config] = None,  # optional: provide backbone.config to copy other hparams
#     ):
#         small_cfg = _mk_reduced_cfg(embed_dim, num_heads, reduction_factor, type, base_cfg)
#         super().__init__(small_cfg, has_relative_attention_bias=has_relative_attention_bias)
#
#         self._side_is_decoder = (type == "decoder")
#         self._stored_memory: Optional[torch.Tensor] = None           # (B, S_enc, d_model_inner) (projected by block internally)
#         self._stored_enc_mask_2d: Optional[torch.Tensor] = None      # (B, S_enc)
#
#         # expose a normalization like your other side blocks expect
#         self.side_norm = nn.LayerNorm(embed_dim)  # outer d_model used by your head
#         self.d_model = embed_dim                  # for shape helpers in your model
#
#     # keep compatibility with your previous API
#     def set_memory(self, memory: torch.Tensor, enc_mask_2d: Optional[torch.Tensor] = None):
#         """
#         memory: encoder hidden states in *outer* width; T5Block will remap internally.
#         enc_mask_2d: encoder padding mask (B, S_enc) with 1 = keep, 0 = pad.
#         """
#         self._stored_memory = memory
#         self._stored_enc_mask_2d = enc_mask_2d
#
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,        # allow (B,S) 2D or (B,1,T,S) 4D
#         position_bias=None,
#         encoder_hidden_states: Optional[torch.Tensor] = None, # optional; if None and decoder, we use stored memory
#         encoder_attention_mask: Optional[torch.Tensor] = None,# allow (B,S_enc) or (B,1,T,S_enc)
#         cross_attn_position_bias=None,
#         past_key_values=None,
#         use_cache: bool = False,
#         output_attentions: bool = False,
#         return_dict: bool = False,
#         cache_position=None,
#         **kwargs,
#     ):
#         # auto-expand 2D masks if needed
#         T = hidden_states.size(1)
#         if (attention_mask is not None) and attention_mask.dim() == 2:
#             attention_mask = _expand_mask_4d(attention_mask, dtype=hidden_states.dtype, tgt_len=T)
#         if (encoder_attention_mask is not None) and encoder_attention_mask.dim() == 2:
#             encoder_attention_mask = _expand_mask_4d(encoder_attention_mask, dtype=hidden_states.dtype, tgt_len=T)
#
#         # if decoder and no memory passed, use stored
#         if self._side_is_decoder and (encoder_hidden_states is None) and (self._stored_memory is not None):
#             encoder_hidden_states = self._stored_memory
#             if encoder_attention_mask is None and self._stored_enc_mask_2d is not None:
#                 encoder_attention_mask = _expand_mask_4d(self._stored_enc_mask_2d, dtype=hidden_states.dtype, tgt_len=T)
#
#         # default cache_position (some HF versions expect it)
#         if cache_position is None:
#             cache_position = torch.arange(T, device=hidden_states.device)
#
#         # call native T5Block forward; keep tuple returns (return_dict=False)
#         try:
#             out = super().forward(
#                 hidden_states,
#                 attention_mask=attention_mask,
#                 position_bias=position_bias,
#                 encoder_hidden_states=encoder_hidden_states,
#                 encoder_attention_mask=encoder_attention_mask,
#                 cross_attn_position_bias=cross_attn_position_bias,
#                 past_key_values=past_key_values,  # new name
#                 use_cache=use_cache,
#                 output_attentions=output_attentions,
#                 return_dict=False,
#                 cache_position=cache_position,
#             )
#         except TypeError:
#             # older transformers fall back
#             out = super().forward(
#                 hidden_states,
#                 attention_mask=attention_mask,
#                 position_bias=position_bias,
#                 encoder_hidden_states=encoder_hidden_states,
#                 encoder_attention_mask=encoder_attention_mask,
#                 cross_attn_position_bias=cross_attn_position_bias,
#                 past_key_value=past_key_values,   # old name
#                 use_cache=use_cache,
#                 output_attentions=output_attentions,
#                 return_dict=False,
#                 cache_position=cache_position,
#             )
#         return out  # (hidden_states, ...) like T5Block

class ReducedT5Block(nn.Module):
    def __init__(
                    self,
                    embed_dim: int,
                    num_heads: int,
                    reduction_factor: int = 1,
                    type: str = "encoder",   # "encoder" | "decoder"
                    has_relative_attention_bias: bool = False,
                    base_cfg: Optional[T5Config] = None,  # optional: provide backbone.config to copy other hparams
                ):
        super().__init__()
        pass


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization — matches LlamaRMSNorm exactly."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(orig_dtype)


class LlamaBlock(nn.Module):
    """
    Reduced-width LLaMA-2-7B-style decoder block for the LST side network (TEE/CPU).
    Uses RMSNorm (matching backbone) and SwiGLU MLP.
    Causal mask generated internally from seq_len.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        reduction_factor: int = 1,
        mlp_ratio: float = 11008 / 4096,  # exact LLaMA-2-7B intermediate ratio
        final_norm: bool = True,
    ):
        super().__init__()
        self.input_shape = None
        self.reduction_factor = reduction_factor
        dim_side = embed_dim // reduction_factor

        # Largest divisor of dim_side ≤ num_heads with head_dim ≥ 32
        num_heads_side = num_heads
        while num_heads_side > 1 and (dim_side % num_heads_side != 0 or dim_side // num_heads_side < 32):
            num_heads_side -= 1
        self.num_heads_side = num_heads_side

        self.norm1 = RMSNorm(dim_side)
        self.attn  = nn.MultiheadAttention(dim_side, num_heads_side, batch_first=True)

        self.norm2     = RMSNorm(dim_side)
        hidden         = max(1, int(dim_side * mlp_ratio))
        self.gate_proj = nn.Linear(dim_side, hidden, bias=False)
        self.up_proj   = nn.Linear(dim_side, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim_side, bias=False)
        self.act       = nn.SiLU()

        if final_norm:
            self.side_norm = RMSNorm(dim_side)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_shape is None:
            self.input_shape = x.shape
        B, S, D = x.shape

        causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        attn_bias = x.new_zeros(S, S)
        attn_bias.masked_fill_(causal, float('-inf'))

        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_bias, need_weights=False)
        x = x + attn_out

        x_norm2 = self.norm2(x)
        x = x + self.down_proj(self.act(self.gate_proj(x_norm2)) * self.up_proj(x_norm2))
        return x

    @property
    def module_str(self):
        return f'LlamaBlock(rf={self.reduction_factor})'

    @property
    def config(self):
        return {'name': LlamaBlock.__name__, 'reduction_factor': self.reduction_factor}

    def get_model_size(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def is_zero_layer():
        return False

    def get_flops(self):
        if self.input_shape is None:
            return 0, None
        B, S, D = self.input_shape
        attn_flops = 4 * linear_flops(D, D, B * S)
        hidden = self.gate_proj.out_features
        mlp_flops = 2 * linear_flops(D, hidden, B * S) + linear_flops(hidden, D, B * S)
        return attn_flops + mlp_flops, self.input_shape


def build_layer_from_config(rf: int, cfg: Dict) -> nn.Module:  # type: ignore[override]
    """Instantiate LAConv2d / LALinear according to ``cfg['type']``."""
    t = cfg["type"].lower()
    cfg.update({"rf": rf})
    if t.startswith("conv2d"):
        return LAConv2d.from_config(cfg)
    if t.startswith("linear"):
        return LALinear.from_config(cfg)
    raise ValueError(f"Unsupported layer type: {cfg['type']}")

# Functional Test
if __name__ == '__main__':

    # ConvLayer
    in_channels = 3
    out_channels = 64
    kernel_size = 3 
    padding = 1
    stride = 1
    dilation = 1
    groups = 1
    bias = True
    has_shuffle = False
    use_bn = True
    act_func = 'relu'
    dropout_rate = 0.0
    conv_layer = ReducedConvLayer(in_channels, out_channels, kernel_size,
                            padding, stride, dilation, groups, bias, has_shuffle, 
                            use_bn, act_func, dropout_rate)
    print(conv_layer)
    print(conv_layer.get_model_size())

    # IdentityLayer
    in_channels = 3
    out_channels = 64
    use_bn = False
    act_func = None
    dropout_rate = 0.0
    identity_layer = EmptyLayer(in_channels, out_channels,
                                   use_bn, act_func, dropout_rate)
    print(identity_layer)
    print(identity_layer.get_model_size())

    # ZeroLayer
    stride = 1
    zero_layer = EmptyLayer(stride) # ZeroLayer
    print(zero_layer)
    print(zero_layer.get_model_size())