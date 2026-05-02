# connectors.py
# -----------------------------------------------------------------------------
# Channel & spatial size matching that preserves memory format (incl. NHWC),
# optimized for TEE latency, and safe for autograd when needed.
#
# - No convs, no permutes, no .contiguous() on the fast (no-grad) path.
# - Zero-copy grouping via as_strided; single reduction with out=...
# - When gradients are required, uses 1x1 convs (autograd-friendly) that
#   naturally propagate channels_last with minimal overhead.
# -----------------------------------------------------------------------------

from __future__ import annotations
import math
import torch
import torch.nn.functional as F


# ============================== Small helpers =============================== #

def _empty_like_channels_same_fmt(x: torch.Tensor, c_out: int) -> torch.Tensor:
    """
    Allocate (N, c_out, H, W) with the *same memory format* as x, without
    relying on Tensor.suggest_memory_format() (for older PyTorch).
    """
    assert x.dim() == 4, "Expected 4D tensor (N, C, H, W)"
    n, _, h, w = x.shape

    want_cl = False
    try:
        # This exists on older versions too
        want_cl = x.is_contiguous(memory_format=torch.channels_last)
    except Exception:
        want_cl = False

    try:
        memfmt = torch.channels_last if want_cl else torch.contiguous_format
        return torch.empty((n, c_out, h, w),
                           dtype=x.dtype, device=x.device,
                           memory_format=memfmt)
    except Exception:
        # Fallback: default allocation
        return torch.empty((n, c_out, h, w), dtype=x.dtype, device=x.device)


def _need_grad(x: torch.Tensor) -> bool:
    # We only incur autograd-safe ops if they are really needed
    return bool(torch.is_grad_enabled() and x.requires_grad)


# ============== Cached 1x1 conv weights for autograd-friendly path ============== #

_avg1x1_cache: dict[tuple, torch.Tensor] = {}
_expand1x1_cache: dict[tuple, torch.Tensor] = {}

def _get_avg1x1(c_out: int, group: int, device, dtype) -> torch.Tensor:
    """
    Weight for grouped 1x1 conv that averages 'group' input channels per output.
    Shape: (c_out, group, 1, 1); groups=c_out; in_channels = c_out * group
    """
    key = ("avg", device, dtype, c_out, group)
    w = _avg1x1_cache.get(key)
    if w is None:
        w = torch.full((c_out, group, 1, 1), 1.0 / group,
                       device=device, dtype=dtype, requires_grad=False)
        _avg1x1_cache[key] = w
    return w


def _get_expand1x1(c: int, c_out: int, device, dtype) -> torch.Tensor:
    """
    Weight for 1x1 conv that replicates channels (one-hot mapping).
    Shape: (c_out, c, 1, 1); groups=1
    """
    key = ("expand", device, dtype, c, c_out)
    w = _expand1x1_cache.get(key)
    if w is None:
        w = torch.zeros((c_out, c, 1, 1),
                        device=device, dtype=dtype, requires_grad=False)
        idx_o = torch.arange(c_out, device=device)
        w[idx_o, idx_o % c, 0, 0] = 1
        _expand1x1_cache[key] = w
    return w


# ============================== Core primitives ============================== #

def _reduce_channels_grouped_fast(
    x: torch.Tensor,
    c_out: int,
    mode: str = "avg",
    remainder: str = "drop",
) -> torch.Tensor:
    """
    Fast no-grad reduction C -> c_out using as_strided + single-kernel reduction
    with out=... (preserves memory format, lowest latency).
    """
    assert x.dim() == 4
    n, c, h, w = x.shape
    assert c_out < c, "Use expand path for C -> larger c_out"

    if remainder != "drop":
        raise NotImplementedError("Lowest-latency path supports remainder='drop' only.")

    group = c // c_out
    if group <= 0:
        raise RuntimeError("Invalid reduction configuration (group <= 0).")

    c_used = group * c_out
    x0 = x[:, :c_used, :, :]  # narrow (view)

    # Valid for both contiguous (NCHW) and channels_last
    sN, sC, sH, sW = x0.stride()

    # Split C_used -> (c_out, group) with a view
    # xg[n, i, g, h, w] = x0[n, i*group + g, h, w]
    xg = x0.as_strided((n, c_out, group, h, w),
                       (sN, sC * group, sC, sH, sW))

    out = _empty_like_channels_same_fmt(x, c_out)

    if mode in ("avg", "mean"):
        torch.sum(xg, dim=2, out=out)  # single reduction kernel
        out.mul_(1.0 / group)
    elif mode == "sum":
        torch.sum(xg, dim=2, out=out)
    elif mode == "max":
        torch.amax(xg, dim=2, out=out)
    else:
        raise ValueError("mode must be in {'avg','mean','sum','max'}")

    return out


def _reduce_channels_grouped_grad(
    x: torch.Tensor,
    c_out: int,
    mode: str = "avg",
    remainder: str = "drop",
) -> torch.Tensor:
    """
    Autograd-friendly reduction C -> c_out via grouped 1x1 conv.
    Preserves channels_last and supports gradient flow to x.
    """
    assert x.dim() == 4
    n, c, h, w = x.shape
    assert c_out < c, "Use expand path for C -> larger c_out"

    if remainder != "drop":
        raise NotImplementedError("For simplicity, use remainder='drop'.")

    group = c // c_out
    if group <= 0:
        raise RuntimeError("Invalid reduction configuration (group <= 0).")

    c_used = group * c_out
    xt = x[:, :c_used, :, :]  # narrow (view)

    if mode in ("avg", "mean"):
        w = _get_avg1x1(c_out, group, xt.device, xt.dtype)
        # groups=c_out ensures each output channel consumes 'group' inputs
        y = F.conv2d(xt, w, bias=None, stride=1, padding=0, groups=c_out)
        return y
    elif mode == "sum":
        # sum == avg * group
        w = _get_avg1x1(c_out, group, xt.device, xt.dtype)
        y = F.conv2d(xt, w, bias=None, stride=1, padding=0, groups=c_out)
        return y.mul_(group)
    elif mode == "max":
        # True max across arbitrary groups is not expressible via a single 1x1 conv.
        # If you need autograd + max, fall back to explicit view+amax (allocates).
        xg = xt.reshape(n, c_out, group, h, w)
        return torch.amax(xg, dim=2)
    else:
        raise ValueError("mode must be in {'avg','mean','sum','max'}")


def _expand_channels_replicate_fast(x: torch.Tensor, c_out: int) -> torch.Tensor:
    """
    Fast no-grad expansion C -> c_out by replication using slice copies
    into a preallocated output that matches input memory format.
    """
    assert x.dim() == 4
    n, c, h, w = x.shape
    out = _empty_like_channels_same_fmt(x, c_out)
    if c == 0:
        return out.zero_()

    full = c_out // c
    rem  = c_out - full * c

    # Copy full channel blocks
    for k in range(full):
        out[:, k * c : (k + 1) * c, :, :] = x

    # Copy remainder channels
    if rem:
        out[:, full * c : full * c + rem, :, :] = x[:, :rem, :, :]

    return out


def _expand_channels_replicate_grad(x: torch.Tensor, c_out: int) -> torch.Tensor:
    """
    Autograd-friendly expansion C -> c_out via 1x1 conv with one-hot weights.
    Preserves channels_last and allows gradients to flow to x.
    """
    assert x.dim() == 4
    n, c, h, w = x.shape
    w = _get_expand1x1(c, c_out, x.device, x.dtype)
    # Standard conv (groups=1) with one-hot weights => replication
    return F.conv2d(x, w, bias=None, stride=1, padding=0)


# ============================ Public entry points ============================ #

def _match_channels_np(
    x: torch.Tensor,
    out_c: int,
    mode: str = "avg",
    *,
    remainder: str = "drop",
) -> torch.Tensor:
    """
    Channel count matcher that preserves memory format (incl. channels_last).

    4D CNN tensors (N,C,H,W):
      - out_c < C: group/pool
          * no-grad: as_strided + reduction(out=...)  [lowest latency]
          * grad:    grouped 1x1 conv                 [autograd safe]
      - out_c > C: expand
          * no-grad: slice-copy                       [lowest latency]
          * grad:    1x1 conv one-hot                 [autograd safe]
      - out_c == C: return x view

    3D ViT tensors (N,L,D): (channels_last not applicable)
      - Shrink: reshape to (N,L,out_c,group) and mean over group.
      - Expand: repeat and slice to out_c.
    """
    dim = x.dim()

    assert dim== 4

    n, c, h, w = x.shape
    if out_c == c:
        return x
    if out_c < c:
        print(f'_match_channels_np: dimension is not the same: {out_c}, {c}')
        if _need_grad(x):
            return _reduce_channels_grouped_grad(x, out_c, mode=mode, remainder=remainder)
        else:
            return _reduce_channels_grouped_fast(x, out_c, mode=mode, remainder=remainder)
    else:
        if _need_grad(x):
            return _expand_channels_replicate_grad(x, out_c)
        else:
            return _expand_channels_replicate_fast(x, out_c)


def _match_spatial_np(
    x: torch.Tensor,
    target_spatial: tuple[int, int] | tuple[int],
    is_vit: bool,
    *,
    up_mode: str = "bilinear",
) -> torch.Tensor:
    """
    Spatial matcher that preserves memory format where applicable.

    CNN (4D): downscale via adaptive_avg_pool2d; upscale via interpolate.
              Both propagate channels_last automatically.
    ViT (3D): treat 'spatial' as sequence length L; operate on (N, D, L).
    """
    if is_vit:
        assert x.dim() == 3, "Expected (N, L, D) for ViT"
        n, l, d = x.shape
        tgt_l = int(target_spatial[0])
        if l == tgt_l:
            return x
        x1 = x.transpose(1, 2)  # (N, D, L)
        if l >= tgt_l:
            x1 = F.adaptive_avg_pool1d(x1, tgt_l)
        else:
            x1 = F.interpolate(x1, size=tgt_l, mode="nearest")
        return x1.transpose(1, 2)

    # CNN path
    assert x.dim() == 4, "Expected 4D (N, C, H, W) for CNN"
    n, c, h, w = x.shape
    tgt_h, tgt_w = map(int, target_spatial)
    if (h, w) == (tgt_h, tgt_w):
        return x
    print(f'_match_spatial_np: dimension is not the same: {(tgt_h, tgt_w)}, {(h, w)}')
    if h >= tgt_h and w >= tgt_w:
        return F.adaptive_avg_pool2d(x, (tgt_h, tgt_w))
    else:
        if up_mode == "bilinear":
            return F.interpolate(x, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
        else:
            return F.interpolate(x, size=(tgt_h, tgt_w), mode="nearest")

def _match_dim_np(
    x: torch.Tensor,
    out_d: int,
    mode: str = "avg",
    *,
    remainder: str = "drop",
) -> torch.Tensor:
    assert x.dim() == 3, f"_match_dim_np expects (N,L,D), got {tuple(x.shape)}"
    N, L, Din = x.shape
    if out_d <= 0:
        raise ValueError(f"out_d must be > 0, got {out_d}")
    if out_d == Din:
        return x

    if out_d < Din:
        # --- shrink: group/pool ---
        if remainder not in {"drop", "pad"}:
            raise ValueError(f"remainder must be 'drop' or 'pad', got {remainder}")

        if remainder == "drop":
            group = Din // out_d
            if group == 0:
                # out_d > Din handled earlier; guard unlikely rounding issue
                group = 1
            usable = out_d * group
            x2 = x[..., :usable]  # drop tail; view-friendly
        else:  # 'pad'
            # ceil to a multiple of out_d
            usable = ((Din + out_d - 1) // out_d) * out_d
            pad = usable - Din
            if pad:
                pad_zeros = torch.zeros((N, L, pad), dtype=x.dtype, device=x.device)
                x2 = torch.cat([x, pad_zeros], dim=-1)
            else:
                x2 = x
            group = usable // out_d

        # reshape then reduce over the 'group' axis
        x3 = x2.reshape(N, L, out_d, group)
        if mode == "avg":
            y = x3.mean(dim=-1)
        elif mode == "sum":
            y = x3.sum(dim=-1)
        elif mode == "max":
            y = x3.amax(dim=-1)
        else:
            raise ValueError(f"unsupported mode '{mode}', choose from 'avg','sum','max'")
        return y

    else:
        # --- expand: repeat then slice ---
        reps = (out_d + Din - 1) // Din  # ceil(out_d / Din)
        y = x.repeat(1, 1, reps)[..., :out_d]
        return y



# =============================== Self-checks ================================ #
if __name__ == "__main__":
    # Basic checks for memory-format preservation & autograd safety.
    for memfmt in ("nchw", "nhwc"):
        x = torch.randn(2, 48, 16, 20)
        if memfmt == "nhwc":
            x = x.to(memory_format=torch.channels_last)
        is_cl = x.is_contiguous(memory_format=torch.channels_last)

        # No-grad fast path
        with torch.no_grad():
            y = _match_channels_np(x, 12, mode="avg")
            assert y.shape == (2, 12, 16, 20)
            assert y.is_contiguous(memory_format=torch.channels_last) == is_cl

            z = _match_channels_np(x, 96, mode="avg")
            assert z.shape == (2, 96, 16, 20)
            assert z.is_contiguous(memory_format=torch.channels_last) == is_cl

        # Grad path
        xg = x.clone().requires_grad_(True)
        yg = _match_channels_np(xg, 12, mode="avg")
        (yg.sum()).backward()  # should not error
        assert yg.is_contiguous(memory_format=torch.channels_last) == is_cl

    # ViT path
    xv = torch.randn(2, 197, 768, requires_grad=True)
    yv = _match_channels_np(xv, 192, mode="avg")
    yv.sum().backward()
    assert yv.shape == (2, 197, 192)

    yv2 = _match_spatial_np(xv.detach(), (128,), is_vit=True)
    assert yv2.shape == (2, 128, 768)
