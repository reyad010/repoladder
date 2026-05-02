"""flops_computation.py
--------------------------------------------------
A lightweight utility module that collects **all standalone helper
functions** used for FLOPs estimation in your code‑base.

Import this file anywhere you need FLOPs maths without dragging in the
full `layers.py` implementation.
"""
from __future__ import annotations

import math
from typing import Tuple
from torch import nn
from fvcore.nn import FlopCountAnalysis, flop_count_table

__all__ = [
    "conv2d_flops",
    "linear_flops",
    "reduced_conv_out",
]

# -----------------------------------------------------------------------------
#   Core FLOPs formulas
# -----------------------------------------------------------------------------

def conv2d_flops(
    C_in: int,
    C_out: int,
    k_h: int,
    k_w: int,
    out_h: int,
    out_w: int,
    groups: int = 1,
) -> int:
    """Compute **multiply‑add FLOPs per sample** for a 2‑D convolution.

    Formula (PyTorch convention):
        MACs = C_out · out_h · out_w · (C_in / groups) · k_h · k_w
        FLOPs = 2 × MACs  (multiply **and** add)
    """
    macs = C_out * out_h * out_w * (C_in // groups) * k_h * k_w
    return 2 * macs


def linear_flops(in_features: int, out_features: int, instances: int) -> int:
    """Compute FLOPs for a Linear layer.

    For a dense matrix multiplication **followed** by the bias add the
    cost per *instance* is 2 · in_features · out_features.
    If the input has *instances* rows (e.g. B×S tokens) the total is:

        FLOPs = 2 · instances · in_features · out_features
    """
    return 2 * instances * in_features * out_features


# -----------------------------------------------------------------------------
#   Small helper shared by Reduced* layers (channel reduction logic only)
# -----------------------------------------------------------------------------

def reduced_conv_out(
    in_c: int,
    out_c: int,
    reduction_factor: int,
    layer_type: str,
) -> Tuple[int, int]:
    """Return (effective_in, effective_out) after channel reduction.

    * **connection**  : keep *input* channels intact; reduce *output*.
    * **side**        : reduce *both* in & out.
    * **upsample**    : reduce *input* only (keeps output full size).
    """
    layer_type = layer_type.lower()
    if layer_type == "connection":
        return in_c, out_c // reduction_factor
    if layer_type == "side":
        return in_c // reduction_factor, out_c // reduction_factor
    if layer_type == "upsample":
        return in_c // reduction_factor, out_c
    raise ValueError(f"invalid layer_type: {layer_type}")


def model_get_flops(self: nn.Module):
    # ── nested helper so that it captures surrounding scope cleanly ──
    def _sum_flops(module: nn.Module):
        total = 0
        for child in module.children():
            if hasattr(child, "get_flops"):
                # Prefer layer‑level declared shape if present
                child_in = getattr(child, "input_shape")
                flp, out_shape = child.get_flops(child_in)
                print(child_in, flp, out_shape)
                total += flp
            elif any(child.children()):
                # Container without explicit FLOPs – recurse into its kids
                flp, out_shape = _sum_flops(child)
                total += flp
            else: total, out_shape = 0, None
            # else: leaf op (e.g. ReLU) with no cost – ignore
        return total, out_shape

    return _sum_flops(self)

# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def add_flops_method(model: nn.Module):
    """Attach :func:`model_get_flops` to an *existing* model in‑place.
    Returns the model for chaining.
    """
    if hasattr(model, "get_flops"):
        raise AttributeError("Model already defines `get_flops`. No action taken.")

    model.get_flops = types.MethodType(model_get_flops, model)  # type: ignore[attr-defined]
    return model

# ---------------------------------------------------------------------------
# End of file
# ---------------------------------------------------------------------------
def compute_flops_with_fvscore(model, dummy_inputs, dummy_features=None, table_print=False):
    import torch
    model.eval()
    with torch.no_grad():
        if dummy_features is None:
            flops = FlopCountAnalysis(model, dummy_inputs)
        else:
            flops = FlopCountAnalysis(model, (dummy_inputs, dummy_features))
    if table_print: print(flop_count_table(flops, max_depth=2))
    total_flops = flops.total()
    avg_flops = total_flops / dummy_inputs.size()[0] * 2
    print(f"Avg FLOPs: {avg_flops / 1e6 :.4f} MFLOPS")
    return avg_flops