"""Fisher-prune init for LST-ViT side network.

Builds a fresh timm vit_base_patch16_224, computes Fisher importance with the
downstream task labels, walks an ordered layer list propagating a single
"kept-index" set in the original 768-dim space (LST paper Sec. 3.2), and
copies the reduced weights into LadderSideTuneNet.side_layers.* using a key
remap to ViTBlock naming (norm1, attn.{in_proj,out_proj}, norm2, mlp.0/3,
side_norm).

ViT-base is constant-width (D=768 across all 12 blocks), so a single keep-set
of size D' = D // rf propagates through every block — the same setup that
works for T5 in unused/st_security/t5_prune.py.

Connection layers (Linear 768 -> 768/rf) and the input ladder are left at
their PyTorch default init in this stage. If the gap doesn't close enough,
they can be initialized as one-hot row-selectors against the per-block kept
sets in a follow-up.
"""
from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn


# ───────────────────────── Fisher importance ──────────────────────────────────
def compute_fisher_vit(
    model: nn.Module, data_loader, device,
    num_batches: int = 10, grad_type: str = "square",
) -> Dict[str, torch.Tensor]:
    """Sum of (squared) gradients over `num_batches` batches.

    The pretrained timm ViT outputs 1000-class ImageNet logits; we compute CE
    against task labels clamped to that class space. The signal is biased by
    the label-space mismatch but still ranks weight importance usefully — what
    matters for init is which directions in weight space have large gradient
    magnitudes for *any* image-classification objective.
    """
    reducer = torch.square if grad_type == "square" else torch.abs
    grad_sums = {n: torch.zeros_like(p, device=device)
                 for n, p in model.named_parameters()}
    model.train()
    model.to(device)
    for i, (imgs, targets) in enumerate(data_loader):
        if i >= num_batches:
            break
        imgs = imgs.to(device)
        targets = targets.to(device)
        logits = model(imgs)
        nc = logits.shape[-1]
        targets = targets.clamp_max(nc - 1)
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                grad_sums[n] += reducer(p.grad.detach())
        for p in model.parameters():
            p.grad = None
    return grad_sums


# ───────────────────────── pruning primitives ─────────────────────────────────
def _l1_prune_idxs(weights: torch.Tensor, amount: float) -> List[int]:
    """Indices along dim=0 with smallest L1 norm; size = round(n*amount)."""
    n = weights.shape[0]
    n_prune = int(round(n * float(amount)))
    if n_prune <= 0:
        return []
    if n_prune >= n:
        return list(range(n))
    flat = weights.detach().abs().reshape(n, -1).sum(dim=1)
    return torch.argsort(flat)[:n_prune].tolist()


def _drop_rows(t: torch.Tensor, drop_idxs: List[int], dim: int = 0) -> torch.Tensor:
    n = t.shape[dim]
    if not drop_idxs:
        return t
    drop_set = set(int(i) for i in drop_idxs)
    keep = [i for i in range(n) if i not in drop_set]
    keep_t = torch.tensor(keep, dtype=torch.long, device=t.device)
    return torch.index_select(t, dim, keep_t).contiguous()


def _drop_cols(t: torch.Tensor, drop_idxs: List[int]) -> torch.Tensor:
    return _drop_rows(t, drop_idxs, dim=1)


def _per_head_qkv_drop(
    qkv_imp: torch.Tensor, amount: float, num_dim: int, num_heads: int,
) -> Tuple[List[int], List[int]]:
    """Per-head L1 prune of QKV importance to preserve attention structure.

    Without this, a global L1 over the 768-dim Q/K/V space could prune all
    indices from one head and none from another. We instead prune each head's
    head_dim slice independently, ensuring each side head keeps `head_dim/rf`
    indices.

    qkv_imp: shape (num_dim, ...) — element-wise product of Q,K,V importance,
             expected after input-dim pruning has already been applied.
    Returns (single_drop, fused_drop):
      - single_drop: drop indices in [0, num_dim) for one Q/K/V tile
      - fused_drop:  drop indices in [0, 3*num_dim) for the stacked qkv weight
    """
    head_dim = num_dim // num_heads
    drop: List[int] = []
    for h in range(num_heads):
        s, e = h * head_dim, (h + 1) * head_dim
        local = _l1_prune_idxs(qkv_imp[s:e], amount)
        drop.extend(s + i for i in local)
    fused = (drop
             + [i + num_dim for i in drop]
             + [i + 2 * num_dim for i in drop])
    return drop, fused


# ───────────────────────── backbone-side prune walk ───────────────────────────
def prune_vit_backbone(
    model: nn.Module, rf: int,
    importance_measure: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Walk ordered layers, propagate one keep-index set through all 12 blocks.

    Returns a state_dict where every backbone Linear/LayerNorm has been
    width-reduced from D=768 to D//rf (and qkv from 3D to 3*(D//rf), MLP hidden
    from 4D to 4*(D//rf)). Keys preserved (timm naming).

    `importance_measure` should be a Fisher dict; if None, falls back to
    |weight| (truncated init, like t5_prune.py).
    """
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if importance_measure is None:
        imp = {k: v.detach().abs() for k, v in sd.items()}
    else:
        imp = {k: importance_measure[k].detach().clone().to(sd[k].device)
               for k in sd if k in importance_measure}

    prune_val = 1.0 - 1.0 / float(rf)
    num_heads = model.blocks[0].attn.num_heads
    n_blocks = len(model.blocks)
    new_sd: Dict[str, torch.Tensor] = {}

    # bootstrap block-input drop set from norm1.weight importance
    pruning_idxs = _l1_prune_idxs(imp["blocks.0.norm1.weight"].unsqueeze(1),
                                  prune_val)

    for i in range(n_blocks):
        # ── norm1 ─────────────────────────────────────────────────────────
        for sub in ("norm1.weight", "norm1.bias"):
            new_sd[f"blocks.{i}.{sub}"] = _drop_rows(
                sd[f"blocks.{i}.{sub}"], pruning_idxs, dim=0)

        # ── attn.qkv: prune input by pruning_idxs, output per-head ─────────
        qkv_w = sd[f"blocks.{i}.attn.qkv.weight"]
        qkv_b = sd[f"blocks.{i}.attn.qkv.bias"]
        qkv_imp = imp[f"blocks.{i}.attn.qkv.weight"]
        D = qkv_w.shape[1]
        qkv_w = _drop_cols(qkv_w, pruning_idxs)
        qkv_imp = _drop_cols(qkv_imp, pruning_idxs)
        # split q/k/v importance and pick per-head kept indices in [0, D)
        q_imp = qkv_imp[:D]
        k_imp = qkv_imp[D:2 * D]
        v_imp = qkv_imp[2 * D:]
        prod = q_imp * k_imp * v_imp                         # (D, D')
        single_drop, fused_drop = _per_head_qkv_drop(
            prod, prune_val, D, num_heads)
        qkv_w = _drop_rows(qkv_w, fused_drop, dim=0)
        qkv_b = _drop_rows(qkv_b, fused_drop, dim=0)
        new_sd[f"blocks.{i}.attn.qkv.weight"] = qkv_w
        new_sd[f"blocks.{i}.attn.qkv.bias"]   = qkv_b

        # ── attn.proj: input pruned by single_drop, output by NEW prune ────
        proj_w = sd[f"blocks.{i}.attn.proj.weight"]
        proj_b = sd[f"blocks.{i}.attn.proj.bias"]
        proj_imp = imp[f"blocks.{i}.attn.proj.weight"]
        proj_w = _drop_cols(proj_w, single_drop)
        proj_imp = _drop_cols(proj_imp, single_drop)
        proj_drop = _l1_prune_idxs(proj_imp, prune_val)      # in [0, D)
        proj_w = _drop_rows(proj_w, proj_drop, dim=0)
        proj_b = _drop_rows(proj_b, proj_drop, dim=0)
        new_sd[f"blocks.{i}.attn.proj.weight"] = proj_w
        new_sd[f"blocks.{i}.attn.proj.bias"]   = proj_b
        pruning_idxs = proj_drop

        # ── norm2 ─────────────────────────────────────────────────────────
        for sub in ("norm2.weight", "norm2.bias"):
            new_sd[f"blocks.{i}.{sub}"] = _drop_rows(
                sd[f"blocks.{i}.{sub}"], pruning_idxs, dim=0)

        # ── mlp.fc1: input pruned by block idxs, output NEW in [0, 4D) ─────
        fc1_w = sd[f"blocks.{i}.mlp.fc1.weight"]
        fc1_b = sd[f"blocks.{i}.mlp.fc1.bias"]
        fc1_imp = imp[f"blocks.{i}.mlp.fc1.weight"]
        fc1_w = _drop_cols(fc1_w, pruning_idxs)
        fc1_imp = _drop_cols(fc1_imp, pruning_idxs)
        fc1_drop = _l1_prune_idxs(fc1_imp, prune_val)        # in [0, 4D)
        fc1_w = _drop_rows(fc1_w, fc1_drop, dim=0)
        fc1_b = _drop_rows(fc1_b, fc1_drop, dim=0)
        new_sd[f"blocks.{i}.mlp.fc1.weight"] = fc1_w
        new_sd[f"blocks.{i}.mlp.fc1.bias"]   = fc1_b
        mlp_hidden_drop = fc1_drop

        # ── mlp.fc2: input pruned by mlp_hidden_drop, output NEW in [0, D) ─
        fc2_w = sd[f"blocks.{i}.mlp.fc2.weight"]
        fc2_b = sd[f"blocks.{i}.mlp.fc2.bias"]
        fc2_imp = imp[f"blocks.{i}.mlp.fc2.weight"]
        fc2_w = _drop_cols(fc2_w, mlp_hidden_drop)
        fc2_imp = _drop_cols(fc2_imp, mlp_hidden_drop)
        fc2_drop = _l1_prune_idxs(fc2_imp, prune_val)        # in [0, D)
        fc2_w = _drop_rows(fc2_w, fc2_drop, dim=0)
        fc2_b = _drop_rows(fc2_b, fc2_drop, dim=0)
        new_sd[f"blocks.{i}.mlp.fc2.weight"] = fc2_w
        new_sd[f"blocks.{i}.mlp.fc2.bias"]   = fc2_b
        pruning_idxs = fc2_drop

    # Final layer norm — used by the LAST side block's side_norm only
    for sub in ("norm.weight", "norm.bias"):
        new_sd[sub] = _drop_rows(sd[sub], pruning_idxs, dim=0)

    return new_sd


# ───────────────────────── key remap → side_layers.* ──────────────────────────
def remap_vit_keys_to_side(
    pruned_sd: Dict[str, torch.Tensor], n_side: int,
) -> Dict[str, torch.Tensor]:
    """blocks.X.* (timm) → side_layers.X.* (current ViTBlock naming).

    Mapping (only emits keys that exist in ViTBlock parameters):
      norm1.{w,b}              → side_layers.X.norm1.{w,b}
      attn.qkv.{w,b}           → side_layers.X.attn.in_proj_{w,b}
      attn.proj.{w,b}          → side_layers.X.attn.out_proj.{w,b}
      norm2.{w,b}              → side_layers.X.norm2.{w,b}
      mlp.fc1.{w,b}            → side_layers.X.mlp.0.{w,b}
      mlp.fc2.{w,b}            → side_layers.X.mlp.3.{w,b}
      norm.{w,b}  (only)       → side_layers.{n_side-1}.side_norm.{w,b}
    """
    out: Dict[str, torch.Tensor] = {}
    for key, val in pruned_sd.items():
        if not key.startswith("blocks."):
            continue
        parts = key.split(".")
        if len(parts) < 4:
            continue
        i = int(parts[1])
        if i >= n_side:
            continue
        sub = ".".join(parts[2:])
        if sub.startswith("norm1.") or sub.startswith("norm2."):
            out[f"side_layers.{i}.{sub}"] = val
        elif sub == "attn.qkv.weight":
            out[f"side_layers.{i}.attn.in_proj_weight"] = val
        elif sub == "attn.qkv.bias":
            out[f"side_layers.{i}.attn.in_proj_bias"] = val
        elif sub == "attn.proj.weight":
            out[f"side_layers.{i}.attn.out_proj.weight"] = val
        elif sub == "attn.proj.bias":
            out[f"side_layers.{i}.attn.out_proj.bias"] = val
        elif sub == "mlp.fc1.weight":
            out[f"side_layers.{i}.mlp.0.weight"] = val
        elif sub == "mlp.fc1.bias":
            out[f"side_layers.{i}.mlp.0.bias"] = val
        elif sub == "mlp.fc2.weight":
            out[f"side_layers.{i}.mlp.3.weight"] = val
        elif sub == "mlp.fc2.bias":
            out[f"side_layers.{i}.mlp.3.bias"] = val
    if n_side > 0 and "norm.weight" in pruned_sd:
        out[f"side_layers.{n_side - 1}.side_norm.weight"] = pruned_sd["norm.weight"]
        out[f"side_layers.{n_side - 1}.side_norm.bias"]   = pruned_sd["norm.bias"]
    return out


# ───────────────────────── public entry point ─────────────────────────────────
def init_side_vit(
    model: nn.Module, rf: int, train_loader, device,
    num_batches: int = 10, weights_path: str = "./models/vit_base_patch16_224.pth",
) -> int:
    """Fisher-prune timm ViT-base @ rf, copy reduced weights into model.side_layers.

    Expects `model` to be a converted LST-ViT normal_net (i.e., the output of
    LadderSideTuneNet.convert_to_normal_net) — its side_layers should be a
    ModuleList of ViTBlock / EmptyLayer.

    Returns the number of side parameters successfully initialized.
    """
    if not hasattr(model, "side_layers"):
        raise ValueError("init_side_vit: model has no `side_layers` attribute.")

    # 1. fresh timm ViT-base + load the same pretrained weights the LST model uses
    backbone = timm.create_model("vit_base_patch16_224", pretrained=False, embed_dim=768)
    backbone.load_state_dict(torch.load(weights_path))
    backbone = backbone.to(device)

    # 2. compute Fisher importance on the downstream-task loader
    fisher = compute_fisher_vit(backbone, train_loader, device,
                                num_batches=num_batches)

    # 3. width-reduce the backbone state_dict at rf
    pruned_sd = prune_vit_backbone(backbone, rf, importance_measure=fisher)

    # 4. remap timm keys → side_layers.X.*
    n_side = len(model.side_layers)
    remapped = remap_vit_keys_to_side(pruned_sd, n_side)

    # 5. selectively load: skip keys absent from target (Empty side blocks) or
    #    with mismatched shape (defensive — should not happen for ViT-base).
    target_sd = model.state_dict()
    matched = dropped = mismatched = 0
    for k, v in remapped.items():
        if k not in target_sd:
            dropped += 1
            continue
        if target_sd[k].shape != v.shape:
            print(f"[init_side_vit] shape mismatch on {k}: "
                  f"target {tuple(target_sd[k].shape)}, src {tuple(v.shape)} — skip")
            mismatched += 1
            continue
        target_sd[k] = v.to(target_sd[k].dtype).to(target_sd[k].device)
        matched += 1
    model.load_state_dict(target_sd, strict=True)

    print(f"[init_side_vit] rf={rf}: matched {matched} side params "
          f"(dropped {dropped} non-target keys, {mismatched} shape mismatches)")

    del backbone, fisher, pruned_sd, remapped
    return matched
