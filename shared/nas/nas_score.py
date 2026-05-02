import torch
import torchvision.models as models
import numpy as np
import random
import torch.nn.functional as F
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Optional, Tuple, List, Dict

def get_batch_jacobian(net, x, target, **model_kwargs):
    net.zero_grad()
    # Integer inputs (NLP token IDs) cannot require gradients; skip requires_grad_ for them.
    # For hook_logdet NASWOT the Jacobian is unused — backward only fires visited_backwards hooks.
    is_int = x.dtype in (torch.int64, torch.int32, torch.int16, torch.int8)
    if not is_int:
        x.requires_grad_(True)
    y = net(x, **model_kwargs)
    out = torch.tensor([0.0])
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach() if (not is_int and x.grad is not None) else torch.zeros(x.shape[0], 1)
    return jacob, target.detach(), y.detach(), out.detach()

def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld

def random_score(jacob, label=None):
    return np.random.normal()

_scores = {
        'hook_logdet': hooklogdet,
        'random': random_score
        }
def get_score_func(score_name):
    return _scores[score_name]

def nas_score_old(network, x, target, score='hook_logdet', maxofn=1, seed=1):
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    batch_size = x.size(0)
    network.K = np.zeros((batch_size, batch_size))

    def counting_forward_hook(module, inp, out):
        try:
            if not module.visited_backwards:
                return
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x = (inp > 0).float()
            K = x @ x.t()
            K2 = (1. - x) @ (1. - x.t())
            network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()
        except:
            pass

    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True

    for name, module in network.named_modules():
        if 'ReLU' in str(type(module)):
            # hooks[name] = module.register_forward_hook(counting_hook)
            module.register_forward_hook(counting_forward_hook)
            module.register_backward_hook(counting_backward_hook)
    s = []
    for j in range(maxofn):
        x2 = torch.clone(x)
        jacobs, labels, y, out = get_batch_jacobian(network, x, target)

        if 'hook_' in score:
            network(x2)
            s.append(get_score_func(score)(network.K, target))
        else:
            s.append(get_score_func(score)(jacobs, labels))
    return np.mean(s)


def counting_forward_hook(module, inp, out):
    try:
        if not hasattr(module, 'visited_backwards') or not module.visited_backwards:
            return
        if isinstance(inp, tuple):
            inp = inp[0]
        inp = inp.view(inp.size(0), -1)
        x = (inp > 0).float()
        K = x @ x.t()
        K2 = (1. - x) @ (1. - x.t())
        if hasattr(module, 'K_accumulate'):
            module.K_accumulate += K.cpu().numpy() + K2.cpu().numpy()
        else:
            module.K_accumulate = K.cpu().numpy() + K2.cpu().numpy()
    except Exception as e:
        print(f"Hook exception: {e}")

def counting_backward_hook(module, inp, out):
    module.visited_backwards = True

def nas_score(network, x, target, score='hook_logdet', maxofn=1, seed=1, **model_kwargs):
    include_modules = ('side_layers', 'connection_layers', 'upsample'),
    # Save RNG states
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_cpu_state = torch.get_rng_state()
    torch_cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    # Set seeds locally
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    hook_handles = []
    batch_size = x.size(0)
    _hook_activations = (torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU)
    for name, module in network.named_modules():
        if 'backbone' in name and isinstance(module, _hook_activations):
            continue
        # if 'connection_layers' in name:
        #     continue
        if isinstance(module, _hook_activations):
            module.K_accumulate = np.zeros((batch_size, batch_size))
            module.visited_backwards = False
            h1 = module.register_forward_hook(counting_forward_hook)
            h2 = module.register_backward_hook(counting_backward_hook)
            hook_handles.extend([h1, h2])
    was_training = network.training
    network.eval()

    try:
        s = []
        for j in range(maxofn):
            x2 = torch.clone(x)
            jacobs, labels, y, out = get_batch_jacobian(network, x, target, **model_kwargs)

            if 'hook_' in score:
                network(x2, **model_kwargs)
                normalize = False
                if normalize:
                    Ks = []
                    for m in network.modules():
                        if hasattr(m, 'K_accumulate'):
                            K = m.K_accumulate
                            denom = np.linalg.norm(K, ord='fro') + 1e-8  # Frobenius norm
                            Ks.append(K / denom)
                    K_total = np.sum(Ks, axis=0) if Ks else None
                else: K_total = sum([m.K_accumulate for m in network.modules() if hasattr(m, 'K_accumulate')])
                s.append(get_score_func(score)(K_total, target))
            else:
                s.append(get_score_func(score)(jacobs, labels))
        return np.mean(s)/batch_size
    finally:
        # Remove all hooks to restore original model for accurate latency evaluation
        for h in hook_handles:h.remove()
        # Restore original RNG states
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_cpu_state)
        if torch_cuda_state is not None:
            torch.cuda.set_rng_state_all(torch_cuda_state)
        if was_training:
            network.train()

# dummy_inputs = torch.rand(1024, 3, 64, 64).to('cuda:0')
# target = torch.rand(1024).to('cuda:0')
# network = models.alexnet(pretrained=False).to('cuda:0')
#
# s = nas_score(network, dummy_inputs, target)
# print(s)


def nas_score_phantom(net=None, x=None, target=None):
    was_training = net.training
    net.eval()
    # Store the gradient
    grads = {}
    for name, param in net.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.clone()

    net.zero_grad()

    x.requires_grad_(True)
    y = net(x)
    temp_y = y.clone()

    y.backward(torch.ones_like(y))

    jacobs = x.grad.detach()

    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

    corrs = np.corrcoef(jacobs)

    # DEBUG
    corrs_ori = copy.deepcopy(corrs)

    if np.isnan(corrs).any():

        # Check if there are any finite and non-NaN values
        if np.isfinite(corrs).any():
            print("------Warning------: NaN values found. Replacing NaNs with the mean of finite values.")
            # Calculate the mean of finite values
            finite_mean = np.nanmean(corrs[np.isfinite(corrs)])
            # Replace NaNs with the calculated mean
            corrs = np.where(np.isnan(corrs), finite_mean, corrs)

            # Store the jacobs and corrs and temp_y into a file
            np.save("jacobs.npy", jacobs)
            np.save("corrs.npy", corrs)
            np.save("corrs_ori.npy", corrs_ori)
            np.save("temp_y.npy", temp_y.detach().cpu().numpy())

        else:
            print("------Warning------: No finite or non-NaN values found. Using default mean:")
            # Provide a default value if no valid entries exist
            finite_mean = 0  # Default value can be adjusted based on the context
            # Replace NaNs with default value
            corrs = np.where(np.isnan(corrs), finite_mean, corrs)

            # Store the jacobs and corrs and temp_y into a file
            np.save("jacobs.npy", jacobs)
            np.save("corrs.npy", corrs)
            np.save("temp_y.npy", temp_y.detach().cpu().numpy())

    elif np.isinf(corrs).any():
        print("------Warning------: Inf values found. Replacing Infs with the max and min of finite values.")
        max_val = np.max(corrs[np.isfinite(corrs)])
        min_val = np.min(corrs[np.isfinite(corrs)])
        corrs = np.where(corrs == np.inf, max_val, corrs)
        corrs = np.where(corrs == -np.inf, min_val, corrs)

    else:
        pass

    v, _ = np.linalg.eig(corrs)
    v = np.where(v <= 0, 1e-5, v)

    k = 1e-5  # Small value to prevent division by zero

    score = -np.sum(np.log(v + k) + 1. / (v + k))

    # Restore the gradient
    for name, param in net.named_parameters():
        if name in grads:
            param.grad = grads[name]

    if was_training:
        net.train()
    return score


########################
# Helper utilities
########################

def _save_grads_and_mode(net):
    was_training = net.training
    net.eval()
    saved_grads = {}
    for name, p in net.named_parameters():
        if p.grad is not None:
            saved_grads[name] = p.grad.clone()
    return was_training, saved_grads


def _restore_grads_and_mode(net, was_training, saved_grads):
    for name, p in net.named_parameters():
        if name in saved_grads:
            p.grad = saved_grads[name]
        else:
            p.grad = None
    if was_training:
        net.train()


def _count_activation_units(net, x, **model_kwargs):
    """
    Approximate NA (activation atomic for DAS) by counting active ReLU/GELU/SiLU units.
    """
    counts = {"active": 0.0, "total": 0.0}
    handles = []

    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        act = (out > 0).float()
        counts["active"] += act.sum().item()
        counts["total"] += act.numel()

    for m in net.modules():
        if isinstance(m, (torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU)):
            handles.append(m.register_forward_hook(hook_fn))

    was_training = net.training
    net.eval()
    with torch.no_grad():
        net(x, **model_kwargs)
    if was_training:
        net.train()

    for h in handles:
        h.remove()

    return counts["active"], counts["total"]


########################
# 1) EPE-NAS (class-aware Jacobian correlation)
########################

def epe_nas_score(net, x, target, eps=1e-5, seed=1, **model_kwargs):
    """
    EPE-NAS style score:
    - Uses per-class correlation matrices of input Jacobians.
    - This follows the spirit of EPE-NAS (Lopes et al.) but is a simplified implementation.
    """
    # Set seeds deterministically (similar style to nas_score)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    was_training, saved_grads = _save_grads_and_mode(net)
    try:
        x_local = x.detach().clone()
        jacob, labels, y, _ = get_batch_jacobian(net, x_local, target, **model_kwargs)
        B = jacob.size(0)
        J = jacob.view(B, -1).detach().cpu().numpy()
        labels_np = labels.view(-1).detach().cpu().numpy()

        scores = []
        for c in np.unique(labels_np):
            idx = np.where(labels_np == c)[0]
            if len(idx) < 2:
                continue
            Jc = J[idx]  # (Nc, D)
            # Center across samples
            Jc = Jc - Jc.mean(axis=0, keepdims=True)
            # Sample-sample covariance & correlation (Nc x Nc)
            Gc = Jc @ Jc.T  # Gram
            diag = np.sqrt(np.clip(np.diag(Gc), a_min=eps, a_max=None))
            denom = np.outer(diag, diag) + eps
            Rc = Gc / denom

            # Use logdet of Rc as scalar (normalized by size)
            eigvals = np.linalg.eigvalsh(Rc + eps * np.eye(len(idx)))
            eigvals = np.clip(eigvals, eps, None)
            e_c = np.sum(np.log(eigvals)) / (len(idx) ** 2)
            scores.append(e_c)

        if len(scores) == 0:
            return 0.0
        return float(np.mean(scores))
    finally:
        _restore_grads_and_mode(net, was_training, saved_grads)


########################
# 2) DAS-like score (distinguishing + activation atomic)
########################

def das_score(net, x, target, lam=1.0, seed=1, **model_kwargs):
    """
    Distinguishing Activation Score (DAS)-style:
      score ≈ NKH - λ * NA_norm
    - NKH: NASWOT-style kernel logdet (via your nas_score)
    - NA_norm: fraction of active units across ReLU/GELU layers
    This is an approximation of DAS built on top of your NASWOT implementation.
    """
    nkh = nas_score(net, x, target, score='hook_logdet', maxofn=1, seed=seed, **model_kwargs)
    active, total = _count_activation_units(net, x, **model_kwargs)
    if total <= 0:
        na_norm = 0.0
    else:
        na_norm = active / float(total)
    return float(nkh - lam * na_norm)


########################
# 3) SynFlow (pruning-style, no labels needed)
########################

def synflow_score(net, x, target=None, **model_kwargs):
    """
    SynFlow-style score:
      sum |w * grad_w| after a single forward with all-ones input.
    Not meaningful for integer NLP inputs; returns 0.0 in that case.
    """
    if x.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
        return 0.0

    device = x.device
    params = [p for p in net.parameters() if p.requires_grad]

    # Save original weights and grads
    was_training, saved_grads = _save_grads_and_mode(net)
    orig_weights = [p.data.clone() for p in params]

    try:
        net.zero_grad()
        # Make weights positive (as in SynFlow)
        for p in params:
            p.data = p.data.abs()

        # Use ones input with same shape as x
        ones = torch.ones_like(x, device=device)
        out = net(ones, **model_kwargs)
        # Scalar output for backward
        torch.sum(out).backward()

        score = 0.0
        for p in params:
            if p.grad is not None:
                score += torch.sum(torch.abs(p * p.grad)).item()
        return float(score)
    finally:
        # Restore original weights
        for p, w in zip(params, orig_weights):
            p.data.copy_(w)
        # Clear our grads and restore user's grads/mode
        net.zero_grad()
        _restore_grads_and_mode(net, was_training, saved_grads)


########################
# 4) GradNorm (Abdelfattah et al.) style
########################

def grad_norm_score(net, x, target, loss_type='cross_entropy', **model_kwargs):
    """
    GradNorm-style zero-cost proxy:
      L = CE or MSE; score = sum of L2 norms of parameter gradients.
    """
    was_training, saved_grads = _save_grads_and_mode(net)
    try:
        net.zero_grad()
        out = net(x, **model_kwargs)
        if loss_type == 'cross_entropy' and out.ndim >= 2:
            loss = F.cross_entropy(out, target)
        else:
            # Fallback: MSE between logits and one-hot labels
            if out.ndim == 1:
                out_vec = out.view(-1, 1)
            else:
                out_vec = out
            num_classes = out_vec.size(1)
            one_hot = F.one_hot(target, num_classes=num_classes).float()
            loss = F.mse_loss(out_vec, one_hot)

        loss.backward()
        total_norm = 0.0
        for p in net.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item()
        return float(total_norm)
    finally:
        net.zero_grad()
        _restore_grads_and_mode(net, was_training, saved_grads)


########################
# 5) ZiCo (inverse coefficient of variation on gradients)
########################

def zico_score(net, x, target, eps=1e-8, max_samples=None, **model_kwargs):
    """
    ZiCo-style score (approximation, memory friendly):
      - For each sample i, compute gradient g_i (flattened over all params).
      - For each parameter dimension j, compute coefficient of variation:
            CV_j = std_i(|g_{ij}|) / (mean_i(|g_{ij}|) + eps)
      - ZiCo ≈ mean_j 1 / (CV_j + eps)

    Implementation details:
      - Uses allow_unused=True so params not touched in the forward get 0-grad.
      - Accumulates sum(|g|) and sum(|g|^2) across samples instead of storing
        the full [B, P] gradient matrix.
    """
    # Optionally subsample the batch
    if max_samples is not None and x.size(0) > max_samples:
        idx = torch.randperm(x.size(0), device=x.device)[:max_samples]
        x_sel = x[idx]
        target_sel = target[idx]
        # slice per-sample kwargs (e.g. attention_mask)
        model_kwargs_sel = {k: v[idx] for k, v in model_kwargs.items()}
    else:
        x_sel, target_sel = x, target
        model_kwargs_sel = model_kwargs

    was_training, saved_grads = _save_grads_and_mode(net)
    try:
        params = [p for p in net.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in params)

        device = x_sel.device
        sum_abs = torch.zeros(n_params, device=device)
        sum_sq  = torch.zeros(n_params, device=device)

        # Precompute flat slices for each param
        slices = []
        offset = 0
        for p in params:
            n = p.numel()
            slices.append((offset, offset + n))
            offset += n

        B_eff = x_sel.size(0)

        for i in range(B_eff):
            net.zero_grad()
            sample_kwargs = {k: v[i:i+1] for k, v in model_kwargs_sel.items()}
            out = net(x_sel[i:i+1], **sample_kwargs)

            if out.ndim >= 2:
                loss = F.cross_entropy(out, target_sel[i:i+1])
            else:
                # Fallback: MSE if logits aren't 2D
                loss = F.mse_loss(out.view(1, -1),
                                  target_sel[i:i+1].float().view(1, -1))

            # Some params may not be used in this forward -> allow_unused=True
            grads = torch.autograd.grad(
                loss,
                params,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )

            g_flat = torch.zeros(n_params, device=device)
            for (start, end), g in zip(slices, grads):
                if g is not None:
                    g_flat[start:end] = g.reshape(-1).abs().detach()

            sum_abs += g_flat
            sum_sq  += g_flat * g_flat

        # Compute mean and std across samples for each param dimension
        mean_abs = sum_abs / float(B_eff)
        mean_sq  = sum_sq / float(B_eff)
        var_abs  = torch.clamp(mean_sq - mean_abs * mean_abs, min=0.0)
        std_abs  = torch.sqrt(var_abs + eps)

        # Filter out degenerate dims
        valid = mean_abs > eps
        if valid.sum() == 0:
            return 0.0

        mean_abs_valid = mean_abs[valid]
        std_abs_valid  = std_abs[valid]

        cv = std_abs_valid / (mean_abs_valid + eps)
        zico_val = torch.mean(1.0 / (cv + eps)).item()
        return float(zico_val)

    finally:
        net.zero_grad()
        _restore_grads_and_mode(net, was_training, saved_grads)


########################
# 6) RBFleX-like last-layer RBF kernel score
########################

def rbflex_score(net, x, target=None, gamma=None, eps=1e-5, **model_kwargs):
    """
    RBFleX-NAS style (simplified):
      - Use RBF kernel on final logits (no extra feature maps).
      - score = logdet(K + eps I) with K_ij = exp(-gamma ||z_i - z_j||^2).
      - gamma chosen as inverse median pairwise distance if not provided.
    """
    was_training, saved_grads = _save_grads_and_mode(net)
    try:
        with torch.no_grad():
            z = net(x, **model_kwargs)
            Z = z.view(z.size(0), -1)  # (B, D)
            diff = Z.unsqueeze(1) - Z.unsqueeze(0)
            dist2 = (diff ** 2).sum(dim=-1)  # (B, B)

            if gamma is None:
                d_flat = dist2.detach().view(-1)
                d_flat = d_flat[d_flat > 0]
                if d_flat.numel() == 0:
                    gamma_val = 1.0
                else:
                    median = d_flat.median().item()
                    gamma_val = 1.0 / (median + eps)
            else:
                gamma_val = float(gamma)

            K = torch.exp(-gamma_val * dist2)
        K_np = K.cpu().numpy()
        eigvals = np.linalg.eigvalsh(K_np + eps * np.eye(K_np.shape[0]))
        eigvals = np.clip(eigvals, eps, None)
        return float(np.sum(np.log(eigvals)))
    finally:
        _restore_grads_and_mode(net, was_training, saved_grads)


########################
# 7) ePADS-like perturbation-aware NASWOT
########################

def epads_like_score(net, x, target, noise_std=0.01, seed=1, **model_kwargs):
    """
    Simple perturbation-aware variant (inspired by ePADS):
      score = NASWOT(x) - NASWOT(x + Gaussian noise)

    Not meaningful for integer NLP inputs; falls back to plain NASWOT in that case.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    base = nas_score(net, x, target, score='hook_logdet', maxofn=1, seed=seed, **model_kwargs)

    # Integer inputs (NLP token IDs) can't have noise added; return base score directly.
    if x.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
        return float(base)

    noise = noise_std * torch.randn_like(x)
    x_noisy = (x + noise).detach()

    noisy = nas_score(net, x_noisy, target, score='hook_logdet', maxofn=1, seed=seed, **model_kwargs)

    return float(base - noisy)



########################
# 8) Simple ensemble / meta-proxy
########################

def ensemble_zc_score(net,
                      x,
                      target,
                      methods=('naswot', 'epe', 'zico'),
                      weights=None,
                      **kwargs):  # kwargs forwarded to every sub-method (e.g. attention_mask)
    """
    Ensemble over multiple proxies:
      - methods: list of method names in _ZC_METHODS
      - weights: optional list of same length; if None, uniform
      - kwargs: forwarded to each method (e.g. eps, seed, max_samples)
    Returns a single scalar; you can wrap this in your NAS manager as another predictor.
    """
    vals = []
    for m in methods:
        if m not in _ZC_METHODS:
            raise ValueError(f"Unknown ZC method '{m}'. Known: {list(_ZC_METHODS.keys())}")
        fn = _ZC_METHODS[m]
        vals.append(float(fn(net, x, target, **kwargs)))

    vals = np.asarray(vals, dtype=np.float32)
    if vals.std() > 0:
        vals = (vals - vals.mean()) / (vals.std() + 1e-8)

    if weights is None:
        weights_arr = np.ones_like(vals, dtype=np.float32)
    else:
        weights_arr = np.asarray(weights, dtype=np.float32)
        if weights_arr.shape != vals.shape:
            raise ValueError("weights must have same length as methods")
    weights_arr = weights_arr / (weights_arr.sum() + 1e-8)

    return float((vals * weights_arr).sum())


########################
# Registry of proxies
########################

def _naswot_wrapper(net, x, target, **kwargs):
    score = kwargs.pop('score', 'hook_logdet')
    maxofn = kwargs.pop('maxofn', 1)
    seed = kwargs.pop('seed', 1)
    return nas_score(net, x, target, score=score, maxofn=maxofn, seed=seed, **kwargs)

def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute Linear CKA between two representation matrices.
    X: (N, D1), Y: (N, D2). Returns scalar in [0, 1].
    """
    X = X.float()
    Y = Y.float()
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    XtX = X @ X.t()
    YtY = Y @ Y.t()

    hsic_xy = (XtX * YtY).sum()
    hsic_xx = (XtX * XtX).sum()
    hsic_yy = (YtY * YtY).sum()

    denom = torch.sqrt(hsic_xx * hsic_yy + eps)
    return float(hsic_xy / denom)

def _extract_with_hooks(net, x) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Extract backbone final and side penultimate representations by hooking
    into the model's ACTUAL forward pass — no re-implementation needed.

    Hooks:
    - Forward hook on the last backbone layer -> backbone final features
    - Forward pre-hook on net.head (nn.Linear) -> side penultimate features
      (this captures exactly what is fed into the classifier)

    Works for VGG, ResNet, ViT because all share:
        backbone_layers -> connection -> side_layers -> upsample -> head
    """
    model_name = getattr(net, 'model_name', '')
    backbone_capture = {}
    side_capture = {}
    handles = []

    # ---- Hook 1: Capture backbone final representation ----
    #
    # We want the output of the deepest backbone layer that represents
    # "backbone's view of the data". For all models this is the last
    # element processed in the backbone forward loop:
    #   ResNet: backbone_layers[:-3] iterated, so backbone_layers[-4] is the last
    #   VGG:    backbone_layers iterated fully, so backbone_layers[-1]
    #   ViT:    backbone_layers (= backbone.blocks) iterated, so [-1]

    if hasattr(net, 'backbone_layers') and len(net.backbone_layers) > 0:
        if model_name == 'resnet':
            # backbone_layers[-3:] are downsample layers, not in the main loop.
            # The main loop runs backbone_layers[:-3], i.e. indices 0..16.
            # The last one that gets a full forward is backbone_layers[-4] = index 16.
            hook_idx = len(net.backbone_layers) - 4
        else:
            # VGG and ViT: all backbone layers are iterated
            hook_idx = len(net.backbone_layers) - 1

        if 0 <= hook_idx < len(net.backbone_layers):
            def _bb_hook(module, inp, out):
                backbone_capture['feat'] = out.detach()

            handles.append(
                net.backbone_layers[hook_idx].register_forward_hook(_bb_hook)
            )

    # ---- Hook 2: Capture side penultimate representation ----
    #
    # We use a forward pre-hook on net.head. The input to head is exactly
    # the side representation after upsample — this is the "side penultimate"
    # regardless of model type. No need to know the internal forward logic.

    if hasattr(net, 'head') and isinstance(net.head, nn.Module):
        def _side_hook(module, inp):
            if isinstance(inp, tuple) and len(inp) > 0:
                side_capture['feat'] = inp[0].detach()

        handles.append(
            net.head.register_forward_pre_hook(_side_hook)
        )

    # ---- Run the model's own forward ----
    try:
        with torch.no_grad():
            _ = net(x)
    except Exception as e:
        for h in handles:
            h.remove()
        print(f"[feat_align] forward failed: {e}")
        return None, None

    for h in handles:
        h.remove()

    # ---- Post-process captured tensors to (B, D) ----
    backbone_final = backbone_capture.get('feat', None)
    side_penultimate = side_capture.get('feat', None)

    if backbone_final is None or side_penultimate is None:
        return None, None

    backbone_final = _flatten_to_2d(backbone_final, model_name)
    side_penultimate = _flatten_to_2d(side_penultimate, model_name)

    return backbone_final, side_penultimate

def _flatten_to_2d(t: torch.Tensor, model_name: str) -> torch.Tensor:
    """Flatten a captured tensor to (B, D)."""
    if t.ndim == 2:
        return t
    if t.ndim == 4:
        # (B, C, H, W) -> (B, C)
        return F.adaptive_avg_pool2d(t, 1).flatten(1)
    if t.ndim == 3:
        # (B, SeqLen, D) -> use CLS token for ViT, mean-pool otherwise
        if model_name == 'vit-base':
            return t[:, 0, :]  # CLS token
        return t.mean(dim=1)
    return t.flatten(1)

def _class_separability(features: torch.Tensor, labels: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Fisher-like class separability:
    ratio = tr(S_between) / (tr(S_within) + eps), normalized via sigmoid.
    """
    B, D = features.shape
    labels_np = labels.cpu().numpy()
    unique_classes = np.unique(labels_np)

    if len(unique_classes) < 2:
        return 0.0

    global_mean = features.mean(dim=0)
    s_between = 0.0
    s_within = 0.0

    for c in unique_classes:
        mask = (labels == c)
        if mask.sum() < 2:
            continue
        class_features = features[mask]
        class_mean = class_features.mean(dim=0)

        diff = class_mean - global_mean
        s_between += mask.sum().item() * (diff * diff).sum().item()

        centered = class_features - class_mean.unsqueeze(0)
        s_within += (centered * centered).sum().item()

    ratio = s_between / (s_within + eps)
    return float(torch.sigmoid(torch.tensor(ratio / 10.0)).item())

def feature_alignment_score(net, x, target, **kwargs):
    """
    CKA(backbone_final, side_penultimate) + class separability of side features.

    Parameters
    ----------
    net : nn.Module
        LST normal net (backbone + side). Must have backbone_layers, head.
    x : Tensor
        Input batch.
    target : Tensor
        Labels.
    alpha : float
        Weight for CKA vs separability (default 0.6).

    Returns
    -------
    float
        Score (higher = better alignment = likely higher fine-tuned accuracy).
    """
    seed = kwargs.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    was_training = net.training
    net.eval()

    saved_grads = {}
    for name, p in net.named_parameters():
        if p.grad is not None:
            saved_grads[name] = p.grad.clone()

    try:
        backbone_final, side_penultimate = _extract_with_hooks(net, x)

        if backbone_final is None or side_penultimate is None:
            return 0.0

        B = backbone_final.shape[0]
        backbone_flat = backbone_final.reshape(B, -1)
        side_flat = side_penultimate.reshape(B, -1)

        cka_score = linear_cka(side_flat, backbone_flat)
        separability = _class_separability(side_flat, target)

        alpha = kwargs.get('alpha', 0.6)
        score = alpha * cka_score + (1 - alpha) * separability

        return float(score)

    finally:
        for name, p in net.named_parameters():
            if name in saved_grads:
                p.grad = saved_grads[name]
            else:
                p.grad = None
        if was_training:
            net.train()

def backbone_feature_cka_score(net, x, target, **kwargs):
    """Pure CKA between backbone final and side penultimate."""
    seed = kwargs.get('seed', 42)
    random.seed(seed)
    torch.manual_seed(seed)

    was_training = net.training
    net.eval()

    saved_grads = {}
    for name, p in net.named_parameters():
        if p.grad is not None:
            saved_grads[name] = p.grad.clone()

    try:
        backbone_final, side_penultimate = _extract_with_hooks(net, x)

        if backbone_final is None or side_penultimate is None:
            return 0.0

        B = backbone_final.shape[0]
        backbone_flat = backbone_final.reshape(B, -1)
        side_flat = side_penultimate.reshape(B, -1)

        return float(linear_cka(side_flat, backbone_flat))

    finally:
        for name, p in net.named_parameters():
            if name in saved_grads:
                p.grad = saved_grads[name]
            else:
                p.grad = None
        if was_training:
            net.train()

########################
# 9) MPTS — Multi-Point Transfer Score (LST-specific)
########################

def mpts_score(net, x, target, **kwargs):
    """
    Multi-Point Transfer Score (LST-specific):
      For each active connection_layer[i], compute CKA between its output
      (backbone feature projected into side space) and the final side
      penultimate (input to head). Averaged across active connections.

    Intuition: rewards architectures where information injected at EVERY
    rung of the ladder survives to the classifier, not just the last one.
    Unlike CKA (final representations only), MPTS is sensitive to which
    rungs are active (regular vs empty) and whether each rung's signal
    propagates through the full side network.
    """
    seed = kwargs.get('seed', 42)
    random.seed(seed)
    torch.manual_seed(seed)

    was_training = net.training
    net.eval()
    saved_grads = {name: p.grad.clone() for name, p in net.named_parameters() if p.grad is not None}

    try:
        conn_outputs = {}
        side_capture = {}
        handles = []

        if hasattr(net, 'connection_layers'):
            for i, layer in enumerate(net.connection_layers):
                def _make_conn_hook(idx):
                    def _hook(module, inp, out):
                        conn_outputs[idx] = out.detach()
                    return _hook
                handles.append(layer.register_forward_hook(_make_conn_hook(i)))

        if hasattr(net, 'head') and isinstance(net.head, nn.Module):
            def _side_hook(module, inp):
                if isinstance(inp, tuple) and len(inp) > 0:
                    side_capture['feat'] = inp[0].detach()
            handles.append(net.head.register_forward_pre_hook(_side_hook))

        try:
            with torch.no_grad():
                _ = net(x)
        except Exception as e:
            print(f"[mpts_score] forward failed: {e}")
            for h in handles:
                h.remove()
            return 0.0

        for h in handles:
            h.remove()

        side_final = side_capture.get('feat', None)
        if side_final is None or not conn_outputs:
            return 0.0

        model_name = getattr(net, 'model_name', '')
        side_flat = _flatten_to_2d(side_final, model_name)

        cka_vals = []
        for i, feat in conn_outputs.items():
            if feat.abs().mean().item() < 1e-6:  # skip EmptyLayer outputs
                continue
            feat_flat = _flatten_to_2d(feat, model_name)
            cka_vals.append(linear_cka(feat_flat, side_flat))

        return float(np.mean(cka_vals)) if cka_vals else 0.0

    finally:
        for name, p in net.named_parameters():
            p.grad = saved_grads.get(name, None)
        if was_training:
            net.train()


########################
# 10) SRD — Side Residual Discriminability (LST-specific)
########################

def srd_score(net, x, target, **kwargs):
    """
    Side Residual Discriminability (LST-specific):
      1. Extract backbone_final and side_penultimate.
      2. Project side_penultimate onto backbone's column space (via SVD).
      3. Compute class separability (Fisher ratio → sigmoid) on the residual.

    Intuition: the residual is what the side adds BEYOND the frozen backbone.
    High SRD = the side architecture captures discriminative variation the
    backbone misses, which is exactly the job of the TEE-side network.
    CKA rewards alignment; SRD rewards complementarity.
    """
    seed = kwargs.get('seed', 42)
    random.seed(seed)
    torch.manual_seed(seed)

    was_training = net.training
    net.eval()
    saved_grads = {name: p.grad.clone() for name, p in net.named_parameters() if p.grad is not None}

    try:
        backbone_final, side_penultimate = _extract_with_hooks(net, x)

        if backbone_final is None or side_penultimate is None:
            return 0.0

        model_name = getattr(net, 'model_name', '')
        B = backbone_final.shape[0]
        B_flat = _flatten_to_2d(backbone_final, model_name).float()
        S_flat = _flatten_to_2d(side_penultimate, model_name).float()

        # Center both matrices
        B_c = B_flat - B_flat.mean(0, keepdim=True)
        S_c = S_flat - S_flat.mean(0, keepdim=True)

        # Project S_c onto backbone's column space using economy SVD
        try:
            U, _, _ = torch.linalg.svd(B_c, full_matrices=False)  # U: (B, k)
            s_proj = U @ (U.t() @ S_c)
        except Exception:
            return 0.0

        s_res = S_c - s_proj  # component of side features orthogonal to backbone

        return _class_separability(s_res, target)

    finally:
        for name, p in net.named_parameters():
            p.grad = saved_grads.get(name, None)
        if was_training:
            net.train()


########################
# 11) CFS — Connection Fidelity Score (LST-specific)
########################

def cfs_score(net, x, target, **kwargs):
    """
    Connection Fidelity Score (LST-specific):
      For each active connection_layer[i], compute CKA between its input
      (raw backbone feature) and its output (projected into side space).
      Averaged across active connections.

    Intuition: connection_layers are low-rank projections (backbone_dim →
    side_dim / reduction_factor). Aggressive reduction loses information.
    High CFS = the projection preserves task-relevant structure, so the
    side network receives a faithful representation of backbone features.
    Directly sensitive to the reduction_factor dimension of the search space.
    """
    seed = kwargs.get('seed', 42)
    random.seed(seed)
    torch.manual_seed(seed)

    was_training = net.training
    net.eval()
    saved_grads = {name: p.grad.clone() for name, p in net.named_parameters() if p.grad is not None}

    try:
        conn_pre = {}
        conn_post = {}
        handles = []

        if hasattr(net, 'connection_layers'):
            for i, layer in enumerate(net.connection_layers):
                def _make_hook(idx):
                    def _hook(module, inp, out):
                        if isinstance(inp, tuple) and len(inp) > 0:
                            conn_pre[idx] = inp[0].detach()
                        conn_post[idx] = out.detach()
                    return _hook
                handles.append(layer.register_forward_hook(_make_hook(i)))

        try:
            with torch.no_grad():
                _ = net(x)
        except Exception as e:
            print(f"[cfs_score] forward failed: {e}")
            for h in handles:
                h.remove()
            return 0.0

        for h in handles:
            h.remove()

        if not conn_pre:
            return 0.0

        model_name = getattr(net, 'model_name', '')
        cka_vals = []
        for i in conn_pre:
            if i not in conn_post:
                continue
            post = conn_post[i]
            if post.abs().mean().item() < 1e-6:  # skip EmptyLayer outputs
                continue
            pre_flat = _flatten_to_2d(conn_pre[i], model_name)
            post_flat = _flatten_to_2d(post, model_name)
            cka_vals.append(linear_cka(pre_flat, post_flat))

        return float(np.mean(cka_vals)) if cka_vals else 0.0

    finally:
        for name, p in net.named_parameters():
            p.grad = saved_grads.get(name, None)
        if was_training:
            net.train()


########################
# 12) GFA — Gradient-Feature Alignment (LST-specific)
########################

def gfa_score(net, x, target, **kwargs):
    """
    Gradient-Feature Alignment (GFA) — LST-specific zero-cost proxy.

    For each active connection_layer[i]:
      f_i = forward output  (what the frozen backbone delivers into the side)
      g_i = dL/d(conn_out)  (what the side network "requests" from the backbone,
                              computed via one CE backward pass)
      score_i = linear_CKA(f_i, g_i)

    CKA is scale-invariant, so the score is not dominated by rf or connection
    count.  High GFA means backbone features are structurally aligned with the
    gradient signal — i.e. the backbone already provides, at init, representations
    that point in the direction the side network needs to move in order to reduce
    the task loss.  Empty connections contribute g=0 and are skipped automatically.

    Unlike CKA / MPTS (static representational similarity at random init), GFA is:
      - Task-directed (uses labels through CE loss)
      - Sensitive to learning dynamics, not just current feature geometry
      - Consistent across rf groups (CKA normalization removes magnitude bias)
    """
    seed = kwargs.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if not hasattr(net, 'connection_layers') or len(net.connection_layers) == 0:
        return 0.0

    was_training, saved_grads = _save_grads_and_mode(net)

    conn_feat = {}
    conn_grad = {}
    handles   = []

    for i, layer in enumerate(net.connection_layers):
        def _make_hooks(idx):
            def _fwd(module, inp, out):
                conn_feat[idx] = out           # keep attached for backward

            def _bwd(module, grad_in, grad_out):
                if grad_out[0] is not None:
                    conn_grad[idx] = grad_out[0].detach()

            return _fwd, _bwd

        fh, bh = _make_hooks(i)
        handles.append(layer.register_forward_hook(fh))
        # register_full_backward_hook gives dL/d(output) reliably even with
        # multiple autograd nodes, avoiding the FutureWarning in older APIs.
        try:
            handles.append(layer.register_full_backward_hook(bh))
        except AttributeError:
            handles.append(layer.register_backward_hook(bh))

    try:
        net.zero_grad()
        out = net(x)
        if out.ndim >= 2:
            loss = F.cross_entropy(out, target)
        else:
            loss = F.mse_loss(out.view(-1), target.float().view(-1))
        loss.backward()
    except Exception as e:
        print(f"[gfa_score] forward/backward failed: {e}")
        for h in handles:
            h.remove()
        _restore_grads_and_mode(net, was_training, saved_grads)
        return 0.0

    for h in handles:
        h.remove()

    model_name = getattr(net, 'model_name', '')
    cka_vals   = []

    for i in conn_feat:
        if i not in conn_grad:
            continue
        f = conn_feat[i].detach()
        g = conn_grad[i]

        # Skip EmptyLayer outputs (near-zero forward output)
        if f.abs().mean().item() < 1e-6 or g.abs().mean().item() < 1e-6:
            continue

        f_flat = _flatten_to_2d(f, model_name)
        g_flat = _flatten_to_2d(g, model_name)

        cka_vals.append(linear_cka(f_flat, g_flat))

    _restore_grads_and_mode(net, was_training, saved_grads)

    return float(np.mean(cka_vals)) if cka_vals else 0.0


def register_feature_alignment_methods(zc_methods_dict: dict):
    zc_methods_dict['feat_align'] = feature_alignment_score
    zc_methods_dict['cka'] = backbone_feature_cka_score
    zc_methods_dict['mpts'] = mpts_score
    zc_methods_dict['srd'] = srd_score
    zc_methods_dict['cfs'] = cfs_score
    zc_methods_dict['gfa'] = gfa_score



_ZC_METHODS = {
    'naswot': _naswot_wrapper, # original NASWOT function: mainly focused on expressivity
    'naswot_p': nas_score_phantom, # modification version of NASWOT in phantom
    'epe': epe_nas_score, # difference with NASWOT: compute Jacobian-based correlations per class
    'das': das_score, # successor of NASWOT: decouples the NASWOT framework into two atomic metrics (distinguishing automic score)
    'synflow': synflow_score, # weight * gradients: focused on the trainability and optimization ability (synaptic saliency scores and proposed a modified version (synflow))
    'gradnorm': grad_norm_score, # l2-norm of gradients
    'zico': zico_score, # trainability: inverse coefficient of variation of gradients across samples (chatgpt)
    'rbflex': rbflex_score, # instead of aggregating layer-wise expressivity, look directly at the geometry in the last layer using an RBF kernel over activations and input features
    'epads': epads_like_score, # successor 2 of NASWOT: perturbation-aware distinguishing score
}

register_feature_alignment_methods(_ZC_METHODS)