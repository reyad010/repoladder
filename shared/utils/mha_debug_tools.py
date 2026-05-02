# mha_debug_tools.py
import re
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.backends.cuda import sdp_kernel

def _check_arg_device(x):
    if x is None:
        return True
    return x.device.type in ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]

def _arg_requires_grad(x):
    return (x is not None) and x.requires_grad

def explain_mha_fastpath(mha: nn.MultiheadAttention,
                         query: Tensor, key: Tensor, value: Tensor,
                         *, key_padding_mask=None, attn_mask=None,
                         need_weights=True, average_attn_weights=True, is_causal=False):
    """
    Return (is_fastpath: bool, reason: str). The checks mirror the PyTorch
    MultiheadAttention.forward() you posted, so the 'reason' matches 'why_not_fast_path'.
    """
    why = ""

    # 1) initial dtype checks on masks (float masks block fastpath)
    if (attn_mask is not None and torch.is_floating_point(attn_mask)) or \
       (key_padding_mask is not None and torch.is_floating_point(key_padding_mask)):
        why = "floating-point masks are not supported for fast path."

    is_batched = (query.dim() == 3)

    # Canonicalize masks exactly like MHA does
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

    # 2) global fastpath toggle
    if not torch.backends.mha.get_fastpath_enabled():
        why = "torch.backends.mha.get_fastpath_enabled() was not True"
    elif not is_batched:
        why = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
    elif (query is not key) or (key is not value):
        why = "non-self attention was used (query, key, and value are not the same Tensor)"
    elif (mha.in_proj_bias is not None) and (query.dtype != mha.in_proj_bias.dtype):
        why = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({mha.in_proj_bias.dtype}) don't match"
    elif mha.in_proj_weight is None:
        why = "in_proj_weight was None"
    elif query.dtype != mha.in_proj_weight.dtype:
        why = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({mha.in_proj_weight.dtype}) don't match"
    elif mha.training:
        why = "training is enabled"
    elif (mha.num_heads % 2) != 0:
        # NOTE: This one surprises many folks — odd num_heads prevents the fastpath.
        why = "self.num_heads is not even"
    elif not mha.batch_first:
        why = "batch_first was not True"
    elif mha.bias_k is not None:
        why = "self.bias_k was not None"
    elif mha.bias_v is not None:
        why = "self.bias_v was not None"
    elif mha.add_zero_attn:
        why = "add_zero_attn was enabled"
    elif not mha._qkv_same_embed_dim:
        why = "_qkv_same_embed_dim was not True"
    elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
        why = "supplying both src_key_padding_mask and src_mask at the same time is not supported with NestedTensor input"
    elif torch.is_autocast_enabled():
        why = "autocast is enabled"

    if why:
        return False, why

    # 3) tensor-arg checks (TorchScript / device / grad)
    tensor_args = (
        query, key, value,
        mha.in_proj_weight, mha.in_proj_bias,
        mha.out_proj.weight, mha.out_proj.bias
    )
    if torch.overrides.has_torch_function(tensor_args):
        return False, "some Tensor argument has_torch_function"
    # make_fx tracing?
    torch_dispatch_mode_stack = torch.utils._python_dispatch._get_current_dispatch_mode_stack()
    if any(type(x) == torch.fx.experimental.proxy_tensor.ProxyTorchDispatchMode for x in torch_dispatch_mode_stack):
        return False, "we are running make_fx tracing"
    if not all(_check_arg_device(x) for x in tensor_args):
        return False, ("some Tensor argument's device is neither one of cpu, cuda or "
                       f"{torch.utils.backend_registration._privateuse1_backend_name}")
    if torch.is_grad_enabled() and any(_arg_requires_grad(x) for x in tensor_args):
        return False, ("grad is enabled and at least one of query or the "
                       "input/output projection weights or biases requires_grad")

    # If we get here, fastpath would be used.
    return True, ""

def instrument_mha(mha: nn.MultiheadAttention):
    """
    Monkey-patch a specific MHA instance so each forward prints:
      - if fastpath is used (or why not)
      - basic SDPA backend enable flags at call time
    Returns a callable 'unpatch()' to restore the original forward.
    """
    orig_forward = mha.forward

    def debug_forward(query, key, value, *args, **kwargs):
        is_fast, reason = explain_mha_fastpath(
            mha, query, key, value,
            key_padding_mask=kwargs.get("key_padding_mask", None),
            attn_mask=kwargs.get("attn_mask", None),
            need_weights=kwargs.get("need_weights", True),
            average_attn_weights=kwargs.get("average_attn_weights", True),
            is_causal=kwargs.get("is_causal", False),
        )

        if is_fast:
            print("[MHA fastpath] HIT: using torch._native_multi_head_attention")

        else:
            print(f"[MHA fastpath] MISS: {reason}")

        # Light SDPA context (this doesn't tell which kernel is chosen, but shows what is allowed)
        try:
            from torch.backends.cuda import sdp_kernel
            print(f"[SDPA] enabled -> flash={sdp_kernel.is_flash_sdp_enabled()}, "
                  f"mem_efficient={sdp_kernel.is_mem_efficient_sdp_enabled()}, "
                  f"math={sdp_kernel.is_math_sdp_enabled()}")
        except Exception:
            pass

        return orig_forward(query, key, value, *args, **kwargs)

    mha.forward = debug_forward

    def unpatch():
        mha.forward = orig_forward

    return unpatch
