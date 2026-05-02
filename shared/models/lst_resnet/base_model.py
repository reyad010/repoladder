# Description: ResNet model definition and forward pass
import time
import collections
import torch
import torch.nn as nn
from utils.connectors import _match_channels_np, _match_spatial_np
from modules.layers import *
from utils.tensor_transfer_channel import send_value, recv_value, send_value_no_transfer, recv_value_no_transfer
from utils.flops_computation import conv2d_flops, linear_flops
try: from shmio import ack, tensor_hash
except Exception as e: ack, tensor_hash = None, None


def make_conv_layer(cfg):
    # Extract Conv2d-specific arguments
    conv_cfg = {k: v for k, v in cfg.items() if
                k in ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'bias']}

    layers = []
    layers.append(nn.Conv2d(**conv_cfg))

    if cfg.get('use_bn', False):
        layers.append(nn.BatchNorm2d(cfg['out_channels']))

    if cfg.get('act_func') == 'relu':
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class ObfuscationBlock(nn.Module):

    def __init__(self, conv):
        super(ObfuscationBlock, self).__init__()
        self.conv = conv

    def forward(self, x):
        if self.conv is None: return None
        if self.conv.is_zero_layer():
            res = x
        else:
            res = self.conv(x)
        return res

    @property
    def module_str(self):
        return '(%s)' % (self.conv.module_str)

    @property
    def config(self):
        return {
            'name': ObfuscationBlock.__name__,
            'conv': self.conv.config,
        }

    def get_model_size(self):
        return self.conv.get_model_size()


class ResNet(nn.Module):

    def __init__(self, layers_dict, num_classes=10):
        super(ResNet, self).__init__()
        self.fake_forward_feature = None
        self.model_name = 'resnet'
        self.separate_device = False

        # Define Side Blocks
        self.connection_layers = nn.ModuleList(layers_dict['connection_layers'])
        self.side_layers = nn.ModuleList(layers_dict['side_layers'])
        self.upsample = layers_dict['upsample']
        self.head = layers_dict['head']
        self.gate_params = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(17)])
        self.temperature = 0.1
        conv_cfg = [{'in_channels': 3, 'out_channels': 64, 'kernel_size': (7, 7), 'padding': (3, 3), 'stride': (2, 2),
                     'bias': False, 'use_bn': True, 'act_func': 'relu'},
                    {'in_channels': 64, 'out_channels': 64, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1),
                     'bias': False, 'use_bn': True, 'act_func': 'relu'},
                    {'in_channels': 64, 'out_channels': 64, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1),
                     'bias': False, 'use_bn': True, 'act_func': None},
                    {'in_channels': 64, 'out_channels': 64, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1),
                     'bias': False, 'use_bn': True, 'act_func': 'relu'},
                    {'in_channels': 64, 'out_channels': 64, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1),
                     'bias': False, 'use_bn': True, 'act_func': None},
                    {'in_channels': 64, 'out_channels': 128, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (2, 2),
                     'bias': False, 'use_bn': True, 'act_func': 'relu'},
                    {'in_channels': 128, 'out_channels': 128, 'kernel_size': (3, 3), 'padding': (1, 1),
                     'stride': (1, 1), 'bias': False, 'use_bn': True, 'act_func': None},
                    {'in_channels': 128, 'out_channels': 128, 'kernel_size': (3, 3), 'padding': (1, 1),
                     'stride': (1, 1), 'bias': False, 'use_bn': True, 'act_func': 'relu'},
                    {'in_channels': 128, 'out_channels': 128, 'kernel_size': (3, 3), 'padding': (1, 1),
                     'stride': (1, 1), 'bias': False, 'use_bn': True, 'act_func': None},
                    {'in_channels': 128, 'out_channels': 256, 'kernel_size': (3, 3), 'padding': (1, 1),
                     'stride': (2, 2), 'bias': False, 'use_bn': True, 'act_func': 'relu'},
                    {'in_channels': 256, 'out_channels': 256, 'kernel_size': (3, 3), 'padding': (1, 1),
                     'stride': (1, 1), 'bias': False, 'use_bn': True, 'act_func': None},
                    {'in_channels': 256, 'out_channels': 256, 'kernel_size': (3, 3), 'padding': (1, 1),
                     'stride': (1, 1), 'bias': False, 'use_bn': True, 'act_func': 'relu'},
                    {'in_channels': 256, 'out_channels': 256, 'kernel_size': (3, 3), 'padding': (1, 1),
                     'stride': (1, 1), 'bias': False, 'use_bn': True, 'act_func': None},
                    {'in_channels': 256, 'out_channels': 512, 'kernel_size': (3, 3), 'padding': (1, 1),
                     'stride': (2, 2), 'bias': False, 'use_bn': True, 'act_func': 'relu'},
                    {'in_channels': 512, 'out_channels': 512, 'kernel_size': (3, 3), 'padding': (1, 1),
                     'stride': (1, 1), 'bias': False, 'use_bn': True, 'act_func': None},
                    {'in_channels': 512, 'out_channels': 512, 'kernel_size': (3, 3), 'padding': (1, 1),
                     'stride': (1, 1), 'bias': False, 'use_bn': True, 'act_func': 'relu'},
                    {'in_channels': 512, 'out_channels': 512, 'kernel_size': (3, 3), 'padding': (1, 1),
                     'stride': (1, 1), 'bias': False, 'use_bn': True, 'act_func': None}]
        downsample_cfg = [
            {'in_channels': 64, 'out_channels': 128, 'kernel_size': (1, 1), 'padding': (0, 0), 'stride': (2, 2),
             'bias': False, 'use_bn': True, 'act_func': None},
            {'in_channels': 128, 'out_channels': 256, 'kernel_size': (1, 1), 'padding': (0, 0), 'stride': (2, 2),
             'bias': False, 'use_bn': True, 'act_func': None},
            {'in_channels': 256, 'out_channels': 512, 'kernel_size': (1, 1), 'padding': (0, 0), 'stride': (2, 2),
             'bias': False, 'use_bn': True, 'act_func': None}]

        # Define Base ResNet18 model
        self.num_classes = num_classes

        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(p=0.5)
        self.MaxPool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.backbone_layers = nn.ModuleList()
        for cfg in conv_cfg + downsample_cfg:
            self.backbone_layers.append(make_conv_layer(cfg))

        self.transfer_type = 'shm'
        self._channels_last = True
        # self.init_weight()
        # print('current state dict')
        # for k, v in self.state_dict().items():
        #     print(k, v.shape)
        # import torchvision
        # print('pytorch version state dict')
        # state_dict = torchvision.models.resnet18(pretrained=True).state_dict()
        # for k, v in state_dict.items():
        #     print(k, v.shape)
        # exit()


    @torch.no_grad()
    def init_weight(self, *, strict: bool = True):
        """
        Initialize ONLY the ResNet18 *backbone_layers* (your handcrafted conv/bn stacks)
        from torchvision's pretrained resnet18.

        Mapping (torchvision -> ours):
          conv1.*                       -> backbone_layers.0.0.weight (and .1.* for bn1)
          layer1.{0,1}.conv{1,2} / bn{1,2} -> backbone_layers.1..4   (each is [conv,bn,(act)])
          layer2.{0,1}.conv{1,2} / bn{1,2} -> backbone_layers.5..8
          layer3.{0,1}.conv{1,2} / bn{1,2} -> backbone_layers.9..12
          layer4.{0,1}.conv{1,2} / bn{1,2} -> backbone_layers.13..16
          layer2.0.downsample.*           -> backbone_layers.17
          layer3.0.downsample.*           -> backbone_layers.18
          layer4.0.downsample.*           -> backbone_layers.19

        We do NOT load torchvision's fc.* into your head.* (different num_classes).
        """
        import torchvision

        tv_sd = torchvision.models.resnet18(pretrained=True).state_dict()
        cur_sd = self.state_dict()
        converted = collections.OrderedDict()

        def map_key(k: str):
            # stem
            if k == "conv1.weight":
                return "backbone_layers.0.0.weight"
            if k.startswith("bn1."):
                return "backbone_layers.0.1." + k.split(".", 1)[1]  # weight/bias/running_*/num_batches_tracked

            # helper for basicblock conv/bn
            # our backbone_layers index layout:
            # 1: l1.0 conv1/bn1
            # 2: l1.0 conv2/bn2
            # 3: l1.1 conv1/bn1
            # 4: l1.1 conv2/bn2
            # 5..8 for layer2, 9..12 for layer3, 13..16 for layer4
            layer_base = {"layer1": 1, "layer2": 5, "layer3": 9, "layer4": 13}
            parts = k.split(".")
            if len(parts) >= 4 and parts[0] in layer_base and parts[2] in ("conv1", "bn1", "conv2", "bn2"):
                layer = parts[0]  # layer{1..4}
                block = int(parts[1])  # 0 or 1
                sub = parts[2]  # conv1/bn1/conv2/bn2
                rest = ".".join(parts[3:])  # weight or bn buffers

                # per-block conv pair index: (conv1/bn1)=0, (conv2/bn2)=1
                pair = 0 if sub.endswith("1") else 1
                op_in_block = 0 if sub.startswith("conv") else 1  # 0=conv, 1=bn

                # each block has 2 conv/bn "units"
                idx = layer_base[layer] + block * 2 + pair
                return f"backbone_layers.{idx}.{op_in_block}.{rest}"

            # downsample blocks exist only on the first block of layers 2/3/4
            # torchvision: layer2.0.downsample.0.weight / .1.{bn stuff}
            if len(parts) >= 5 and parts[0] in ("layer2", "layer3", "layer4") and parts[1] == "0" and parts[
                2] == "downsample":
                ds_map = {"layer2": 17, "layer3": 18, "layer4": 19}
                idx = ds_map[parts[0]]
                ds_sub = parts[3]  # "0" (conv) or "1" (bn)
                rest = ".".join(parts[4:])

                if ds_sub == "0":
                    return f"backbone_layers.{idx}.0.{rest}"  # conv
                if ds_sub == "1":
                    return f"backbone_layers.{idx}.1.{rest}"  # bn
                return None

            # never load fc.* (classification head mismatch)
            if k.startswith("fc."):
                return None

            return None

        # perform mapping  shape-safe copy
        for k, v in tv_sd.items():
            new_k = map_key(k)
            if new_k is None:
                continue
            if new_k not in cur_sd:
                continue

            if tuple(cur_sd[new_k].shape) != tuple(v.shape):
                print(
                    f"skip initialize layer: {new_k} (from {k}), "
                    f"torchvision shape: {tuple(v.shape)}, cur model shape: {tuple(cur_sd[new_k].shape)}"
                )
                continue

            ref = cur_sd[new_k]
            converted[new_k] = v.to(dtype=ref.dtype, device=ref.device).contiguous()

        cur_sd.update(converted)
        self.load_state_dict(cur_sd, strict=strict)

        print(f"ResNet-LST backbone initialized from torchvision ResNet18 ({len(converted)} tensors loaded)")
        return converted

    @torch.no_grad()
    def init_side(
            self,
            *,
            calc_loader=None,
            fisher_num_samples: int = 1024,
            backbone_shift: int = 1,
            verbose: bool = True,
    ):
        """Initialize *active* side convolutions from pretrained backbone weights.

        This implements the LST (Ladder Side-Tuning) initialization used in your
        previous train.py workflow (pruning_model_vgg  optional Fisher importance),
        but runs directly inside the VGGNet class so you don't need to
        re-convert / rebuild the model per config.

        Parameters
        ----------
        importance_measure:
            Optional dict of per-parameter importance tensors (e.g., Fisher).
            If None, weight magnitude is used.
        calc_loader:
            If provided and fisher_mode != 'no', Fisher importance will be computed
            via compute_fisher_cv(model, loader, num_samples) from fisher.py.
        fisher_mode:
            'no' (default), 'origin', or 'improved'. Passed through to fisher.py if supported.
        fisher_num_samples:
            Number of samples used for Fisher computation.
        fisher_device:
            Device used to compute Fisher (defaults to the device of the first backbone weight).
        backbone_shift:
            Mapping rule from side-layer index i to backbone conv index j.
            LST paper uses side_i <- backbone_{i1}, so default=1.
            If your implementation uses side_i <- backbone_i, set backbone_shift=0.
        """

        # -------- helpers --------
        def _first_param_device():
            for p in self.parameters():
                return p.device
            return torch.device("cpu")

        def _as_importance_tensor(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
            return t.to(device=ref.device, dtype=ref.dtype)

        def _find_active_conv(m) -> nn.Conv2d:
            """Try to locate the active nn.Conv2d inside a side layer block."""
            if isinstance(m, nn.Conv2d):
                return m
            # common pattern: block.conv is Conv2d
            if hasattr(m, "conv") and isinstance(getattr(m, "conv"), nn.Conv2d):
                return getattr(m, "conv")

            # ObfuscationBlock -> MixedEdge_NoParams -> candidate_ops[active_index]
            if hasattr(m, "conv"):
                conv = getattr(m, "conv")
                # Some MixedEdge implementations expose `active_index` / `active_op`.
                if hasattr(conv, "active_op"):
                    op = conv.active_op
                elif hasattr(conv, "active_index") and hasattr(conv, "candidate_ops"):
                    ai = conv.active_index
                    if isinstance(ai, (list, tuple)):
                        ai = ai[0]
                    op = conv.candidate_ops[int(ai)]
                elif hasattr(conv, "candidate_ops") and hasattr(conv, "get_active_op"):
                    op = conv.get_active_op()
                else:
                    op = None

                if op is not None:
                    # ReducedConvLayer(conv=Conv2d, activation=...)
                    if isinstance(op, nn.Conv2d):
                        return op
                    if hasattr(op, "conv") and isinstance(getattr(op, "conv"), nn.Conv2d):
                        return getattr(op, "conv")
                    # last resort: scan children
                    for sub in op.modules():
                        if isinstance(sub, nn.Conv2d):
                            return sub

            # last resort: scan layer children
            for sub in m.modules():
                if isinstance(sub, nn.Conv2d):
                    return sub
            return None

        def _channel_importance(x: torch.Tensor, dim: int) -> torch.Tensor:
            """L1 importance per channel along `dim`."""
            # move `dim` to front
            x2 = x.abs().transpose(0, dim).contiguous()
            return x2.view(x2.size(0), -1).sum(dim=1)

        def _select_indices(scores: torch.Tensor, target: int) -> list[int]:
            """Select indices to KEEP (largest scores), supporting target > len(scores) by repeating."""
            n = int(scores.numel())
            target = int(target)
            if target <= 0:
                return []
            if target <= n:
                keep = torch.topk(scores, k=target, largest=True).indices
                keep = torch.sort(keep).values
                return keep.tolist()
            # need to extend: repeat all channels then add the top remainder
            integer_part = target // n
            frac = target % n
            extend = torch.arange(n, device=scores.device).repeat(integer_part)
            if frac == 0:
                return extend.tolist()
            keep_frac = torch.topk(scores, k=frac, largest=True).indices
            keep_frac = torch.sort(keep_frac).values
            return torch.cat([extend, keep_frac], dim=0).tolist()

        def _index_select_dim(x: torch.Tensor, dim: int, idxs: list[int]) -> torch.Tensor:
            if len(idxs) == 0:
                raise ValueError("empty idxs")
            idx_t = torch.tensor(idxs, device=x.device, dtype=torch.long)
            return torch.index_select(x, dim, idx_t)

        from st_security.pruning.fisher import compute_fisher_cv
        tv_model = torchvision.models.vgg16(pretrained=True).to('cuda:0')
        # NOTE: compute_fisher_cv signature in your fisher.py is (model, data_loader, num_samples, verbose)
        importance_measure = compute_fisher_cv(tv_model, calc_loader, fisher_num_samples)


        # -------- build a list of active target convs --------
        target_convs: list[nn.Conv2d | None] = []

        group1, group2, group3, group4 = [], [], [], []

        for name, param in self.side_layers:
            print(name, param.shape)
        exit()

        for blk in self.side_layers:
            group1.append(blk.conv.candidate_ops)

        for blk in self.side_layers:
            target_convs.append(_find_active_conv(blk))

        # if there are no convs, nothing to do
        if all(c is None for c in target_convs):
            if verbose:
                print("VGGLST/VGGNet init_side: no active Conv2d found in side_layers; skip")
            return

        # Mapping side idx -> backbone idx
        n_back = len(self.backbone_layers)

        def _src_idx(i: int) -> int:
            j = i
            int(backbone_shift)
            if j < 0:
                j = 0
            if j >= n_back:
                j = n_back - 1
            return j

        # torchvision conv indices used in init_weight
        tv_conv_idx = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]

        def _get_importance_for_backbone(j: int) -> torch.Tensor:
            Wref = self.backbone_layers[j].weight
            if importance_measure is None:
                return Wref.detach()
            if isinstance(importance_measure, dict):
                # common key patterns
                candidates = [
                    f"backbone_layers.{j}.weight",
                    f"{j}.conv.weight",
                ]
                if 0 <= j < len(tv_conv_idx):
                    candidates.extend([
                        f"features.{tv_conv_idx[j]}.weight",
                        f"features.{tv_conv_idx[j]}.conv.weight",
                    ])
                for k in candidates:
                    if k in importance_measure:
                        return _as_importance_tensor(importance_measure[k], Wref)
            # fallback
            return Wref.detach()

        # -------- structured pruning / channel selection chain --------
        select_out_idxs: list[int] = None
        inited = 0
        skipped = 0

        for i, tgt in enumerate(target_convs):
            if tgt is None:
                skipped = 1
                continue

            j = _src_idx(i)
            src = self.backbone_layers[j]
            W = src.weight.detach()
            I = _get_importance_for_backbone(j).detach()

            # target shapes
            out2, in2, kH2, kW2 = tgt.weight.shape
            out1, in1, kH1, kW1 = W.shape

            if (kH1, kW1) != (kH2, kW2):
                if verbose:
                    print(f"init_side: skip layer {i} (kernel mismatch src={W.shape} tgt={tgt.weight.shape})")
                skipped = 1
                continue

            # Step 1) prune/select input channels
            if select_out_idxs is None:
                # first conv in the chain: select among src input channels to match tgt in_channels
                scores_in = _channel_importance(I, dim=1)  # per-input-channel score
                keep_in = _select_indices(scores_in, in2)
            else:
                keep_in = select_out_idxs

            if len(keep_in) != in2:
                # allow mismatch only when extending (keep_in may include repeats)
                if verbose:
                    print(f"init_side: warning layer {i} input keep len={len(keep_in)} target_in={in2}")

            W2 = _index_select_dim(W, dim=1, idxs=keep_in)
            I2 = _index_select_dim(I, dim=1, idxs=keep_in)

            # Step 2) prune/select output channels
            scores_out = _channel_importance(I2, dim=0)
            keep_out = _select_indices(scores_out, out2)
            W3 = _index_select_dim(W2, dim=0, idxs=keep_out)

            # bias
            if (src.bias is not None) and (tgt.bias is not None):
                b3 = _index_select_dim(src.bias.detach(), dim=0, idxs=keep_out)
            else:
                b3 = None

            # copy into target conv
            if tuple(W3.shape) != tuple(tgt.weight.shape):
                if verbose:
                    print(
                        f"init_side: skip layer {i} (shape mismatch after select: got {tuple(W3.shape)} want {tuple(tgt.weight.shape)})")
                skipped = 1
                continue

            tgt.weight.copy_(W3.to(dtype=tgt.weight.dtype, device=tgt.weight.device).contiguous())
            if b3 is not None:
                tgt.bias.copy_(b3.to(dtype=tgt.bias.dtype, device=tgt.bias.device).contiguous())

            select_out_idxs = keep_out
            inited = 1

        if verbose:
            print(f"VGGLST/VGGNet init_side: initialized {inited} side convs (skipped {skipped})")

    def forward(self, x=None, verbose=False, return_timeline=False, **kwargs):
        # backbone forward
        timeline = {}
        backbone_features = []
        chlast = self._channels_last
        x = x if x is not None else self.get_dummy_inputs(device='cuda:0',channels_last=chlast)  # x if x is not None else
        x_side = x if not self.separate_device else x.to('cpu', non_blocking=True)
        torch.cuda.synchronize()
        x_backbone = x
        if return_timeline: timeline["backbone_start"] = time.time_ns()
        for i, layer in enumerate(self.backbone_layers[:-3]):
            key = f"f{i}"
            if return_timeline: timeline[f"backbone_{key}_start"] = time.time_ns()
            # print(x_backbone.shape)
            x_backbone = layer(x_backbone)

            if return_timeline: timeline[f"backbone_{key}_end"] = time.time_ns()
            # Prepare features for side connections

            if isinstance(self.connection_layers[i], EmptyLayer):
                backbone_features.append(None)
            else:
                # Record async transfer with CUDA event
                if self.separate_device:
                    if return_timeline: timeline[f"backbone_{key}_trans0"] = time.time_ns()
                    event = torch.cuda.Event(blocking=False)
                    x_backbone_cpu = x_backbone.to(self.side_device, non_blocking=True)
                    event.record()  # mark copy completion
                    if return_timeline: timeline[f"backbone_{key}_trans1"] = time.time_ns()
                    backbone_features.append((x_backbone_cpu, event))  # event
                else:
                    backbone_features.append((x_backbone, None))

            if return_timeline: timeline[f"backbone_{key}_pooling0"] = time.time_ns()
            if i == 0: x_backbone = self.MaxPool2d(x_backbone)
            if return_timeline: timeline[f"backbone_{key}_pooling1"] = time.time_ns()


            # residual layers
            if i in [2, 4, 6, 8, 10, 12, 14, 16]:
                if return_timeline: timeline[f'backbone_{key}_radd0'] = time.time_ns()
                x_backbone = x_backbone + residual_backbone
                x_backbone = self.relu(x_backbone)
                if return_timeline: timeline[f'backbone_{key}_radd1'] = time.time_ns()

            if i in [0, 2, 4, 6, 8, 10, 12, 14]:
                if i in [4, 8, 12]:
                    if return_timeline: timeline[f'backbone_{key}_res0'] = time.time_ns()
                    residual_backbone = self.backbone_layers[int(i / 4) - 4](x_backbone)
                    if return_timeline: timeline[f'backbone_{key}_res1'] = time.time_ns()
                else:
                    residual_backbone = x_backbone

        # side forward
        if return_timeline: timeline["side_start"] = time.time_ns()
        if return_timeline: timeline['side_input_ladder_start'] = timeline['side_start']
        x_side = self.connection_layers[-1](x_side)
        if return_timeline: timeline['side_input_ladder_end'] = time.time_ns()
        for i, layer in enumerate(self.side_layers[:-3]):
            # print(f'side layer {i}')
            key = f"f{i}"
            layer_name = f'f{i}'
            conn = self.connection_layers[i]
            timeline[f'side_{key}_start'] = time.time_ns()
            if not isinstance(conn, EmptyLayer):
                feat, evt = backbone_features[i]
                if evt is not None: evt.synchronize()  # wait for its specific copy only
                if return_timeline: timeline[f"side_{key}_con0"] = time.time_ns()

                reduced_feature = conn(feat)
                # print(reduced_feature.shape)
                if return_timeline: timeline[f"side_{key}_con1"] = time.time_ns()

                # Addition reduced feature
                if return_timeline: timeline[f"side_{key}_add0"] = time.time_ns()
                # reduced_feature = self._apply_connector(reduced_feature, x_side, getattr(self, 'layer_con_type', 'pool'), is_vit=False)
                # ui = torch.sigmoid(self.gate_params[i] / self.temperature)
                # print(x_side.shape)
                # x_side = ui * x_side + (1 - ui) * reduced_feature
                ui = torch.sigmoid(self.gate_params[i] / self.temperature)
                x_side = torch.lerp(x_side, reduced_feature, 1.0 - ui)
                if return_timeline: timeline[f"side_{key}_add1"] = time.time_ns()

            if i == 0:
                timeline[f'side_{key}_pooling0'] = time.time_ns()
                x_side = self.MaxPool2d(x_side)
                timeline[f'side_{key}_pooling1'] = time.time_ns()

            # residual layers
            if i in [2, 4, 6, 8, 10, 12, 14, 16]:
                # print(f'addition {i}')
                if return_timeline: timeline[f'side_{key}_radd0'] = time.time_ns()
                # residual_side = self._apply_connector(residual_side, x_side, getattr(self, 'layer_con_type', 'pool'), is_vit=False)
                x_side = x_side + residual_side
                x_side = self.relu(x_side)
                if return_timeline: timeline[f'side_{key}_radd1'] = time.time_ns()

            if i in [0, 2, 4, 6, 8, 10, 12, 14]:
                if i in [4, 8, 12]:
                    # print(f'downsample layer {i}')
                    if return_timeline: timeline[f'side_{key}_res0'] = time.time_ns()
                    # x_side = self._coerce_x_to_layer_input(x_side, self.side_layers[int(i / 4) - 4])
                    residual_side = self.side_layers[int(i / 4) - 4](x_side)
                    if return_timeline: timeline[f'side_{key}_res1'] = time.time_ns()
                else:
                    residual_side = x_side

            if return_timeline: timeline[f"side_{key}_layer0"] = time.time_ns()
            # x_side = self._coerce_x_to_layer_input(x_side, layer)
            x_side = layer(x_side)
            if return_timeline: timeline[f"side_{key}_layer1"] = time.time_ns()
            if return_timeline: timeline[f'side_{key}_end'] = timeline[f'side_{key}_layer1']

        # head
        if return_timeline: timeline['side_head0'] = time.time_ns()
        x_side = self.avgpool(x_side)
        x_side = torch.flatten(x_side, 1)
        # x_side = self.dropout(x_side)
        x_side = self.upsample(x_side)
        # x_side = self.relu(x_side)
        # x_side = self.dropout(x_side)
        x_side = self.head(x_side)
        if return_timeline: timeline['side_head1'] = time.time_ns()
        if return_timeline: timeline["side_end"] = time.time_ns()
        if return_timeline: return x_side, timeline
        return x_side

    def side_device_init(self, side_device, blocking=True):

        self.blocking = blocking
        self.side_device = side_device
        for layer in self.side_layers:
            layer.to(side_device)
        for layer in self.connection_layers:
            layer.to(side_device)
        self.upsample.to(side_device)
        self.head.to(side_device)
        self.gate_params.to(side_device)
        self.separate_device = True

    # Map trained weights to backbone model
    def map_weights(self, state_dict):
        key_in_state_dict = ['conv1.weight', 'bn1.weight', 'bn1.bias', 'layer1.0.conv1.weight', 'layer1.0.bn1.weight',
                             'layer1.0.bn1.bias', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias',
                             'layer1.1.conv1.weight', 'layer1.1.bn1.weight', 'layer1.1.bn1.bias', 'layer1.1.conv2.weight',
                             'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer2.0.conv1.weight', 'layer2.0.bn1.weight',
                             'layer2.0.bn1.bias', 'layer2.0.conv2.weight', 'layer2.0.bn2.weight', 'layer2.0.bn2.bias',
                             'layer2.0.downsample.0.weight', 'layer2.0.downsample.1.weight', 'layer2.0.downsample.1.bias',
                             'layer2.1.conv1.weight', 'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.conv2.weight',
                             'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer3.0.conv1.weight', 'layer3.0.bn1.weight',
                             'layer3.0.bn1.bias', 'layer3.0.conv2.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias',
                             'layer3.0.downsample.0.weight', 'layer3.0.downsample.1.weight', 'layer3.0.downsample.1.bias',
                             'layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.conv2.weight',
                             'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer4.0.conv1.weight', 'layer4.0.bn1.weight',
                             'layer4.0.bn1.bias', 'layer4.0.conv2.weight', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias',
                             'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.weight', 'layer4.0.downsample.1.bias',
                             'layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.conv2.weight',
                             'layer4.1.bn2.weight', 'layer4.1.bn2.bias',] #  'fc.weight', 'fc.bias'
        key_no_downsample = [key for key in key_in_state_dict if 'downsample' not in key]
        key_downsample = [key for key in key_in_state_dict if 'downsample' in key]
        key_in_state_dict_order = key_no_downsample + key_downsample
        key_in_cur_model = ['backbone_layers.0.0.weight', 'backbone_layers.0.1.weight', 'backbone_layers.0.1.bias', 'backbone_layers.1.0.weight',
                            'backbone_layers.1.1.weight', 'backbone_layers.1.1.bias', 'backbone_layers.2.0.weight', 'backbone_layers.2.1.weight',
                            'backbone_layers.2.1.bias', 'backbone_layers.3.0.weight', 'backbone_layers.3.1.weight', 'backbone_layers.3.1.bias',
                            'backbone_layers.4.0.weight', 'backbone_layers.4.1.weight', 'backbone_layers.4.1.bias', 'backbone_layers.5.0.weight',
                            'backbone_layers.5.1.weight', 'backbone_layers.5.1.bias', 'backbone_layers.6.0.weight', 'backbone_layers.6.1.weight',
                            'backbone_layers.6.1.bias', 'backbone_layers.7.0.weight', 'backbone_layers.7.1.weight', 'backbone_layers.7.1.bias',
                            'backbone_layers.8.0.weight', 'backbone_layers.8.1.weight', 'backbone_layers.8.1.bias', 'backbone_layers.9.0.weight',
                            'backbone_layers.9.1.weight', 'backbone_layers.9.1.bias', 'backbone_layers.10.0.weight', 'backbone_layers.10.1.weight',
                            'backbone_layers.10.1.bias', 'backbone_layers.11.0.weight', 'backbone_layers.11.1.weight', 'backbone_layers.11.1.bias',
                            'backbone_layers.12.0.weight', 'backbone_layers.12.1.weight', 'backbone_layers.12.1.bias', 'backbone_layers.13.0.weight',
                            'backbone_layers.13.1.weight', 'backbone_layers.13.1.bias', 'backbone_layers.14.0.weight', 'backbone_layers.14.1.weight',
                            'backbone_layers.14.1.bias', 'backbone_layers.15.0.weight', 'backbone_layers.15.1.weight', 'backbone_layers.15.1.bias',
                            'backbone_layers.16.0.weight', 'backbone_layers.16.1.weight', 'backbone_layers.16.1.bias', 'backbone_layers.17.0.weight',
                            'backbone_layers.17.1.weight', 'backbone_layers.17.1.bias', 'backbone_layers.18.0.weight', 'backbone_layers.18.1.weight',
                            'backbone_layers.18.1.bias', 'backbone_layers.19.0.weight', 'backbone_layers.19.1.weight', 'backbone_layers.19.1.bias']
        new_state_dict = OrderedDict()
        for k1, k2 in zip(key_in_state_dict_order, key_in_cur_model):
            new_state_dict[k2] = state_dict[k1]

        self.load_state_dict(new_state_dict, strict=False)
        print('resnet model initialized')

    # Initialize all weights = backbone weights  obfuscation block weights
    def init_model(self, state_dict):

        if state_dict is not None:
            # Initialize backbone weights
            self.map_weights(state_dict)
        else:
            raise ValueError('Benign State_dict is None')

    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = True

        # Freeze the backbone weights
        for name, param in self.named_parameters():
            if 'connection_layers' not in name and 'side_layers' not in name and 'head' not in name and 'upsample' not in name and 'gate_params' not in name:
                if param.requires_grad:
                    param.requires_grad = False

        # Freeze the BN running mean and variance
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def freeze_backbone(self):
        self.freeze_params()

    # ---------------- Static profile (params/FLOPs/transfer) ----------------
    def get_static_info(self) -> dict:
        """Return a static-information dict for this *materialized* LST ResNet.

        FLOPs are accounted at 1 MAC = 2 FLOPs (multiply + add) using
        `conv2d_flops`/`linear_flops`. BatchNorm/ReLU/Pool ops are treated
        as free. Computed analytically from module attributes; no inference
        is run. The method is non-destructive: requires_grad flags are saved
        and restored. Only operates on the materialized normal net (concrete
        ReducedConvLayer / ReducedLinearLayer / EmptyLayer instances), not
        the LadderSideTuneNet super-net.
        """
        # ---- save/restore requires_grad ----
        _saved_rg = [p.requires_grad for p in self.parameters()]
        try:
            self.freeze_params()

            B = 100  # dummy batch size

            # ===== Param counts =====
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            trainable_pct = (100.0 * trainable_params / total_params) if total_params > 0 else 0.0
            model_params_M = total_params / 1e6

            # ===== Backbone FLOPs (full ResNet18 forward, B=100) =====
            # spatial dims per main backbone conv (input H,W -> output H,W)
            # conv0: 64x64 -> 32x32 (s=2). pool: 32->16
            # conv1..4: 16x16 (s=1). conv5: 16->8 (s=2). conv6..8: 8x8 (s=1).
            # conv9: 8->4 (s=2). conv10..12: 4x4 (s=1).
            # conv13: 4->2 (s=2). conv14..16: 2x2 (s=1).
            main_in_hw = [
                (64, 64),
                (16, 16), (16, 16), (16, 16), (16, 16),
                (16, 16), (8, 8),  (8, 8),  (8, 8),
                (8, 8),  (4, 4),  (4, 4),  (4, 4),
                (4, 4),  (2, 2),  (2, 2),  (2, 2),
            ]
            main_out_hw = [
                (32, 32),
                (16, 16), (16, 16), (16, 16), (16, 16),
                (8, 8),  (8, 8),  (8, 8),  (8, 8),
                (4, 4),  (4, 4),  (4, 4),  (4, 4),
                (2, 2),  (2, 2),  (2, 2),  (2, 2),
            ]
            ds_in_hw  = [(16, 16), (8, 8), (4, 4)]
            ds_out_hw = [(8, 8),   (4, 4), (2, 2)]

            backbone_finetune_flops = 0
            # 17 main convs
            for i, layer in enumerate(self.backbone_layers[:17]):
                conv = layer[0]  # nn.Conv2d
                kH, kW = conv.kernel_size if isinstance(conv.kernel_size, (tuple, list)) else (conv.kernel_size,) * 2
                Ho, Wo = main_out_hw[i]
                backbone_finetune_flops += B * conv2d_flops(
                    conv.in_channels, conv.out_channels, kH, kW, Ho, Wo
                )
            # 3 downsample 1x1 convs
            for j, layer in enumerate(self.backbone_layers[17:20]):
                conv = layer[0]
                kH, kW = conv.kernel_size if isinstance(conv.kernel_size, (tuple, list)) else (conv.kernel_size,) * 2
                Ho, Wo = ds_out_hw[j]
                backbone_finetune_flops += B * conv2d_flops(
                    conv.in_channels, conv.out_channels, kH, kW, Ho, Wo
                )
            # head Linear(512, num_classes)
            backbone_finetune_flops += linear_flops(512, self.num_classes, B)

            # ===== Online adapter FLOPs (LST side) =====
            def _layer_flops(layer, H_in, W_in):
                """FLOPs for a Reduced*Layer / EmptyLayer, given input H,W (or N for linear)."""
                if layer is None or isinstance(layer, EmptyLayer):
                    return 0
                if isinstance(layer, ReducedConvLayer):
                    kH, kW = layer.kernel_size if isinstance(layer.kernel_size, (tuple, list)) else (layer.kernel_size,) * 2
                    sH, sW = layer.stride if isinstance(layer.stride, (tuple, list)) else (layer.stride,) * 2
                    pH, pW = layer.padding if isinstance(layer.padding, (tuple, list)) else (layer.padding,) * 2
                    dH = layer.dilation if not isinstance(layer.dilation, (tuple, list)) else layer.dilation[0]
                    dW = layer.dilation if not isinstance(layer.dilation, (tuple, list)) else layer.dilation[1]
                    H_out = (H_in + 2 * pH - dH * (kH - 1) - 1) // sH + 1
                    W_out = (W_in + 2 * pW - dW * (kW - 1) - 1) // sW + 1
                    return B * conv2d_flops(
                        layer.in_channels, layer.out_channels, kH, kW, H_out, W_out, layer.groups,
                    )
                if isinstance(layer, ReducedLinearLayer):
                    return linear_flops(layer.in_channels, layer.out_channels, B)
                # Unknown layer type — try generic param-based estimate via get_flops if possible
                return 0

            # Connection layers (i = 0..16): input shape comes from dummy_features[i]
            # (these are the BACKBONE feature shapes feeding the connection convs).
            feat_shapes = [
                (100, 64, 32, 32),
                (100, 64, 16, 16), (100, 64, 16, 16), (100, 64, 16, 16), (100, 64, 16, 16),
                (100, 128, 8, 8),  (100, 128, 8, 8),  (100, 128, 8, 8),  (100, 128, 8, 8),
                (100, 256, 4, 4),  (100, 256, 4, 4),  (100, 256, 4, 4),  (100, 256, 4, 4),
                (100, 512, 2, 2),  (100, 512, 2, 2),  (100, 512, 2, 2),  (100, 512, 2, 2),
            ]
            online_adapter_flops = 0

            # Active connection layers (skip empty)
            for i in range(17):
                conn = self.connection_layers[i]
                if isinstance(conn, EmptyLayer):
                    continue
                _, _, H, W = feat_shapes[i]
                online_adapter_flops += _layer_flops(conn, H, W)

            # Input ladder = connection_layers[-1] (index 17), input is (100, 3, 64, 64)
            input_ladder = self.connection_layers[-1]
            online_adapter_flops += _layer_flops(input_ladder, 64, 64)

            # Side layers 0..16: input H,W = dummy_features[i] (side stream lives at same spatial)
            for i in range(17):
                side = self.side_layers[i]
                _, _, H, W = feat_shapes[i]
                online_adapter_flops += _layer_flops(side, H, W)

            # Side downsample 1x1 stride-2 convs at side_layers[17, 18, 19] applied at i=4,8,12.
            # input H,W for them = feat_shapes[4], feat_shapes[8], feat_shapes[12]
            for k, src_idx in enumerate([4, 8, 12]):
                side_ds = self.side_layers[17 + k]
                _, _, H, W = feat_shapes[src_idx]
                online_adapter_flops += _layer_flops(side_ds, H, W)

            # Upsample (ReducedLinearLayer) + head (nn.Linear)
            online_adapter_flops += _layer_flops(self.upsample, 1, 1)  # H/W unused for linear
            if isinstance(self.head, nn.Linear):
                online_adapter_flops += linear_flops(
                    self.head.in_features, self.head.out_features, B
                )

            online_adapter_flops_pct = (
                100.0 * online_adapter_flops / backbone_finetune_flops
            ) if backbone_finetune_flops > 0 else 0.0

            # ===== Mask / transfer accounting =====
            # LST: no mask, all transfers are GPU -> TEE.
            # Transfer set: input 'x' + every active connection feature 'f{i}'.
            transfer_shapes = {'x': (B, 3, 64, 64)}
            for i in range(17):
                if not isinstance(self.connection_layers[i], EmptyLayer):
                    transfer_shapes[f'f{i}'] = feat_shapes[i]

            feature_transfer_count = len(transfer_shapes)
            feature_transfer_bytes = 0
            for shp in transfer_shapes.values():
                n = 1
                for d in shp:
                    n *= int(d)
                feature_transfer_bytes += n * 4  # fp32

            online_mask_flops = 0
            offline_mask_flops = 0

            return {
                "model_name": "resnet",
                "peft": "lst",
                "trainable_params": int(trainable_params),
                "total_params": int(total_params),
                "trainable_pct": float(trainable_pct),
                "model_params_M": float(model_params_M),
                "online_adapter_flops": int(online_adapter_flops),
                "backbone_finetune_flops": int(backbone_finetune_flops),
                "online_adapter_flops_pct": float(online_adapter_flops_pct),
                "online_mask_flops": int(online_mask_flops),
                "offline_mask_flops": int(offline_mask_flops),
                "feature_transfer_bytes": int(feature_transfer_bytes),
                "feature_transfer_count": int(feature_transfer_count),
            }
        finally:
            for p, rg in zip(self.parameters(), _saved_rg):
                p.requires_grad = rg

    def get_dummy_inputs(self, device='cpu', channels_last=True):
        if channels_last:
            dummy_inputs = (torch.randn(100, 3, 64, 64).contiguous(memory_format=torch.channels_last).to(device))
        else:
            dummy_inputs = (torch.randn(100, 3, 64, 64).to(device))
        return dummy_inputs

    def get_dummy_features(self, device='cpu', channels_last=True):
        if not channels_last:
            dummy_features = [
                torch.rand((100, 64, 32, 32)).to(device),
                torch.rand((100, 64, 16, 16)).to(device),
                torch.rand((100, 64, 16, 16)).to(device),
                torch.rand((100, 64, 16, 16)).to(device),
                torch.rand((100, 64, 16, 16)).to(device),
                torch.rand((100, 128, 8, 8)).to(device),
                torch.rand((100, 128, 8, 8)).to(device),
                torch.rand((100, 128, 8, 8)).to(device),
                torch.rand((100, 128, 8, 8)).to(device),
                torch.rand((100, 256, 4, 4)).to(device),
                torch.rand((100, 256, 4, 4)).to(device),
                torch.rand((100, 256, 4, 4)).to(device),
                torch.rand((100, 256, 4, 4)).to(device),
                torch.rand((100, 512, 2, 2)).to(device),
                torch.rand((100, 512, 2, 2)).to(device),
                torch.rand((100, 512, 2, 2)).to(device),
                torch.rand((100, 512, 2, 2)).to(device),
            ]
        else:
            dummy_features = [
                torch.rand((100, 64, 32, 32)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 64, 16, 16)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 64, 16, 16)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 64, 16, 16)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 64, 16, 16)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 128, 8, 8)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 128, 8, 8)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 128, 8, 8)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 128, 8, 8)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 256, 4, 4)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 256, 4, 4)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 256, 4, 4)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 256, 4, 4)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 512, 2, 2)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 512, 2, 2)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 512, 2, 2)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 512, 2, 2)).contiguous(memory_format=torch.channels_last).to(device),
            ]
        if hasattr(self, "connection_layers"):
            for i, layer in enumerate(self.connection_layers):
                if isinstance(layer, EmptyLayer):
                    dummy_features[i] = None
        elif hasattr(self, "output_feature_flag"):
            for i, flag in self.output_feature_flag:
                if not flag:
                    dummy_features[i] = None

        return dummy_features

    def convert_to_backbone_model(self, device, real_transfer=True):
        # build flags indicating if we need current layer's fature map
        self.output_feature_flag = []
        for connection_layer in self.connection_layers:
            self.output_feature_flag.append(not isinstance(connection_layer, EmptyLayer))

        # get transfer tensor's name and shape
        shapes = {'x': tuple(self.get_dummy_inputs(device='cpu', channels_last=True).shape)}
        feats = self.get_dummy_features(device='cpu', channels_last=True) or []
        num_feat = len(feats)  # backbone-aligned features only (no input-ladder)

        # get transfer tensor's name and shape
        for i, conn in enumerate(self.connection_layers):
            # Skip input-ladder (and any extra trailing connections beyond feats)
            if i >= num_feat: continue
            # Only add a slot if this connection is active and the feature exists
            if (not isinstance(conn, EmptyLayer)) and feats[i] is not None:
                shapes[f"f{i}"] = tuple(feats[i].shape)
        self.transfer_shapes = shapes

        del self.side_layers, self.connection_layers, self.gate_params, self.upsample, self.head
        self._channels_last = True  # backbone stream uses channels_last
        self._real_transfer = bool(real_transfer)
        self.forward = self.backbone_forward_eager_mode if self._real_transfer else self.backbone_forward_eager_mode_no_transfer_latency
        for layer in self.backbone_layers: layer.to(device)

    def convert_to_side_model(self, device, eager_mode=False, real_transfer=True):
        del self.backbone_layers
        self.side_device_init(device)

        # predefine transfer shapes so recv_value(type='tensor') can reshape
        shapes = {'x': tuple(self.get_dummy_inputs(device='cpu', channels_last=True).shape)}

        feats = self.get_dummy_features(device='cpu', channels_last=True) or []
        num_feat = len(feats)  # backbone-aligned features only (no input-ladder)

        for i, conn in enumerate(self.connection_layers):
            # Skip input-ladder (and any extra trailing connections beyond feats)
            if i >= num_feat: continue
            # Only add a slot if this connection is active and the feature exists
            if (not isinstance(conn, EmptyLayer)) and feats[i] is not None:
                shapes[f"f{i}"] = tuple(feats[i].shape)

        self.transfer_shapes = shapes
        self._channels_last = True
        self._real_transfer = bool(real_transfer)
        self.forward = self.side_forward_eager_mode if self._real_transfer else self.side_forward_eager_mode_no_transfer_latency

    def backbone_forward_eager_mode(self, shm_tensors, iter, verbose=False, inputs=None):
        # backbone forward
        timeline = {}
        check_value = '0'
        chlast = self._channels_last
        x = inputs if inputs is not None else self.get_dummy_inputs(device='cuda:0', channels_last=chlast)
        send_names = [f'x']
        send_value(f'x', x, 'tensor', method=self.transfer_type, shared_memory=shm_tensors['x'], channels_last=chlast, check_value=check_value)
        timeline['backbone_start'] = time.time_ns()
        for i, layer in enumerate(self.backbone_layers[:-3]):
            key = f"f{i}"
            timeline[f'backbone_{key}_start'] = time.time_ns()
            timeline[f'backbone_{key}_layer0'] = time.time_ns()
            x = layer(x)
            torch.cuda.synchronize()
            timeline[f'backbone_{key}_layer1'] = time.time_ns()
            if hasattr(self, "output_feature_flag") and self.output_feature_flag[i]:
                timeline[f'backbone_{key}_trans0'] = time.time_ns()
                if verbose: print(f'{key}: {tensor_hash(x)[-5:]}')
                send_value(key, x, 'tensor', method=self.transfer_type, shared_memory=shm_tensors[key], channels_last=chlast, check_value=check_value)
                timeline[f'backbone_{key}_trans1'] = time.time_ns()
                send_names.append(key)
            if i == 0:
                timeline[f'backbone_{key}_pooling0'] = time.time_ns()
                x = self.MaxPool2d(x)
                torch.cuda.synchronize()
                timeline[f'backbone_{key}_pooling1'] = time.time_ns()

            # residual layers
            if i in [2, 4, 6, 8, 10, 12, 14, 16]:
                timeline[f'backbone_{key}_radd0'] = time.time_ns()
                x = x + residual_backbone
                x = self.relu(x)
                torch.cuda.synchronize()
                timeline[f'backbone_{key}_radd1'] = time.time_ns()

            if i in [0, 2, 4, 6, 8, 10, 12, 14]:
                if i in [4, 8, 12]:
                    timeline[f'backbone_{key}_res0'] = time.time_ns()
                    residual_backbone = self.backbone_layers[int(i / 4) - 4](x)
                    torch.cuda.synchronize()
                    timeline[f'backbone_{key}_res1'] = time.time_ns()
                else:
                    residual_backbone = x

            timeline[f'backbone_{key}_end'] = time.time_ns()

        timeline[f'backbone_end'] = time.time_ns()

        return timeline

    def side_forward_eager_mode(self, iter, verbose=False):
        timeline = {}
        chlast = getattr(self, "_channels_last", True)
        shapes = self.transfer_shapes
        check_value = '0' if iter in [11] else '0'

        # 1) recv input x
        timeline['side_x_recv0'] = time.time_ns()
        x_side_in = recv_value(f'x', 'tensor', shape=tuple(shapes['x']),method=self.transfer_type, channels_last=chlast, check_value=check_value)
        timeline['side_x_recv1'] = time.time_ns()
        # print(f'iter {iter}: channel_last: {x_side_in.is_contiguous(memory_format=torch.channels_last)}, regular_channel: {x_side_in.is_contiguous()}')
        timeline['side_start'] = time.time_ns()

        timeline['side_input_ladder_start'] = timeline['side_start']
        x_side = self.connection_layers[-1](x_side_in)
        timeline['side_input_ladder_end'] = time.time_ns()
        recv_names = [f'x']
        for i, layer in enumerate(self.side_layers[:-3]):
            key = f"f{i}"
            layer_name = f'f{i}'
            conn = self.connection_layers[i]
            timeline[f'side_{key}_start'] = time.time_ns()
            if not isinstance(conn, EmptyLayer) and (layer_name in shapes):
                timeline[f'side_{key}_recv0'] = time.time_ns()
                feat_i = recv_value(key, 'tensor', shape=tuple(shapes[layer_name]),method=self.transfer_type, channels_last=chlast, check_value=check_value)
                timeline[f'side_{key}_recv1'] = time.time_ns()

                timeline[f'side_{key}_con0'] = time.time_ns()
                reduced_feature = conn(feat_i)  # feat_i dummy_features[i]
                if verbose: print(f'{key}: {tensor_hash(feat_i)[-5:]}')
                timeline[f'side_{key}_con1'] = time.time_ns()
                recv_names.append(key)

                # Addition: connection feature  side feature
                timeline[f'side_{key}_add0'] = time.time_ns()
                # reduced_feature = self._apply_connector(reduced_feature, x_side, getattr(self, 'layer_con_type', 'pool'), is_vit=False)
                # ui = torch.sigmoid(self.gate_params[i] / self.temperature)
                # x_side = ui * x_side + (1 - ui) * reduced_feature
                ui = torch.sigmoid(self.gate_params[i] / self.temperature)
                x_side.lerp_(reduced_feature, 1.0 - ui)
                timeline[f'side_{key}_add1']  = time.time_ns()

            if i == 0:
                timeline[f'side_{key}_pooling0'] = time.time_ns()
                x_side = self.MaxPool2d(x_side)
                timeline[f'side_{key}_pooling1'] = time.time_ns()

            # residual layers
            if i in [2, 4, 6, 8, 10, 12, 14, 16]:
                timeline[f'side_{key}_radd0'] = time.time_ns()
                # residual_side = self._apply_connector(residual_side, x_side, getattr(self, 'layer_con_type', 'pool'), is_vit=False)
                x_side = x_side + residual_side
                x_side = self.relu(x_side)
                timeline[f'side_{key}_radd1'] = time.time_ns()


            if i in [0, 2, 4, 6, 8, 10, 12, 14]:
                if i in [4, 8, 12]:
                    # timeline[f'side_{key}_residual0'] = time.time_ns()
                    timeline[f'side_{key}_res0'] = time.time_ns()
                    # x_side = self._coerce_x_to_layer_input(x_side, self.side_layers[int(i / 4) - 4])
                    residual_side = self.side_layers[int(i / 4) - 4](x_side)
                    timeline[f'side_{key}_res1'] = time.time_ns()
                    # timeline[f'side_{key}_residual1'] = time.time_ns()

                else:
                    residual_side = x_side
            timeline[f'side_{key}_layer0'] = time.time_ns()
            # x_side = self._coerce_x_to_layer_input(x_side, layer)
            x_side = layer(x_side)
            timeline[f'side_{key}_layer1'] = time.time_ns()
            timeline[f'side_{key}_end'] = timeline[f'side_{key}_layer1']



        for name in recv_names:
            ack(name)


        # head
        timeline['side_head0'] = time.time_ns()
        x_side = self.avgpool(x_side)
        x_side = torch.flatten(x_side, 1)
        # x_side = self.dropout(x_side)
        x_side = self.upsample(x_side)
        # x_side = self.relu(x_side)
        # x_side = self.dropout(x_side)
        x_side = self.head(x_side)
        timeline['side_head1'] = time.time_ns()
        timeline[f'side_end'] = timeline['side_head1']
        return x_side, timeline

    def backbone_forward_eager_mode_no_transfer_latency(self, shm_tensors, iter, verbose=False, inputs=None):
        # Identical to backbone_forward_eager_mode but uses send_value_no_transfer for tensors so
        # only the publish handshake fires; the actual D->H copy is skipped. Compare A vs A_hat.
        timeline = {}
        check_value = '0'
        chlast = self._channels_last
        x = inputs if inputs is not None else self.get_dummy_inputs(device='cuda:0', channels_last=chlast)
        send_names = [f'x']
        send_value_no_transfer(f'x', x, 'tensor', method=self.transfer_type, shared_memory=shm_tensors['x'], channels_last=chlast, check_value=check_value)
        timeline['backbone_start'] = time.time_ns()
        for i, layer in enumerate(self.backbone_layers[:-3]):
            key = f"f{i}"
            timeline[f'backbone_{key}_start'] = time.time_ns()
            timeline[f'backbone_{key}_layer0'] = time.time_ns()
            x = layer(x)
            torch.cuda.synchronize()
            timeline[f'backbone_{key}_layer1'] = time.time_ns()
            if hasattr(self, "output_feature_flag") and self.output_feature_flag[i]:
                timeline[f'backbone_{key}_trans0'] = time.time_ns()
                if verbose: print(f'{key}: {tensor_hash(x)[-5:]}')
                send_value_no_transfer(key, x, 'tensor', method=self.transfer_type, shared_memory=shm_tensors[key], channels_last=chlast, check_value=check_value)
                timeline[f'backbone_{key}_trans1'] = time.time_ns()
                send_names.append(key)
            if i == 0:
                timeline[f'backbone_{key}_pooling0'] = time.time_ns()
                x = self.MaxPool2d(x)
                torch.cuda.synchronize()
                timeline[f'backbone_{key}_pooling1'] = time.time_ns()

            if i in [2, 4, 6, 8, 10, 12, 14, 16]:
                timeline[f'backbone_{key}_radd0'] = time.time_ns()
                x = x + residual_backbone
                x = self.relu(x)
                torch.cuda.synchronize()
                timeline[f'backbone_{key}_radd1'] = time.time_ns()

            if i in [0, 2, 4, 6, 8, 10, 12, 14]:
                if i in [4, 8, 12]:
                    timeline[f'backbone_{key}_res0'] = time.time_ns()
                    residual_backbone = self.backbone_layers[int(i / 4) - 4](x)
                    torch.cuda.synchronize()
                    timeline[f'backbone_{key}_res1'] = time.time_ns()
                else:
                    residual_backbone = x

            timeline[f'backbone_{key}_end'] = time.time_ns()

        timeline[f'backbone_end'] = time.time_ns()
        return timeline

    def side_forward_eager_mode_no_transfer_latency(self, iter, verbose=False):
        # Identical to side_forward_eager_mode but uses recv_value_no_transfer for tensors so
        # the publish handshake still synchronizes but no real data is consumed. Returned
        # tensors are zeros of the expected shape; outputs are not numerically meaningful.
        timeline = {}
        chlast = getattr(self, "_channels_last", True)
        shapes = self.transfer_shapes
        check_value = '0' if iter in [11] else '0'

        timeline['side_x_recv0'] = time.time_ns()
        x_side_in = recv_value_no_transfer(f'x', 'tensor', shape=tuple(shapes['x']), method=self.transfer_type, channels_last=chlast, check_value=check_value)
        timeline['side_x_recv1'] = time.time_ns()
        timeline['side_start'] = time.time_ns()

        timeline['side_input_ladder_start'] = timeline['side_start']
        x_side = self.connection_layers[-1](x_side_in)
        timeline['side_input_ladder_end'] = time.time_ns()
        recv_names = [f'x']
        for i, layer in enumerate(self.side_layers[:-3]):
            key = f"f{i}"
            layer_name = f'f{i}'
            conn = self.connection_layers[i]
            timeline[f'side_{key}_start'] = time.time_ns()
            if not isinstance(conn, EmptyLayer) and (layer_name in shapes):
                timeline[f'side_{key}_recv0'] = time.time_ns()
                feat_i = recv_value_no_transfer(key, 'tensor', shape=tuple(shapes[layer_name]), method=self.transfer_type, channels_last=chlast, check_value=check_value)
                timeline[f'side_{key}_recv1'] = time.time_ns()

                timeline[f'side_{key}_con0'] = time.time_ns()
                reduced_feature = conn(feat_i)
                if verbose: print(f'{key}: {tensor_hash(feat_i)[-5:]}')
                timeline[f'side_{key}_con1'] = time.time_ns()
                recv_names.append(key)

                timeline[f'side_{key}_add0'] = time.time_ns()
                ui = torch.sigmoid(self.gate_params[i] / self.temperature)
                x_side.lerp_(reduced_feature, 1.0 - ui)
                timeline[f'side_{key}_add1']  = time.time_ns()

            if i == 0:
                timeline[f'side_{key}_pooling0'] = time.time_ns()
                x_side = self.MaxPool2d(x_side)
                timeline[f'side_{key}_pooling1'] = time.time_ns()

            if i in [2, 4, 6, 8, 10, 12, 14, 16]:
                timeline[f'side_{key}_radd0'] = time.time_ns()
                x_side = x_side + residual_side
                x_side = self.relu(x_side)
                timeline[f'side_{key}_radd1'] = time.time_ns()

            if i in [0, 2, 4, 6, 8, 10, 12, 14]:
                if i in [4, 8, 12]:
                    timeline[f'side_{key}_res0'] = time.time_ns()
                    residual_side = self.side_layers[int(i / 4) - 4](x_side)
                    timeline[f'side_{key}_res1'] = time.time_ns()
                else:
                    residual_side = x_side
            timeline[f'side_{key}_layer0'] = time.time_ns()
            x_side = layer(x_side)
            timeline[f'side_{key}_layer1'] = time.time_ns()
            timeline[f'side_{key}_end'] = timeline[f'side_{key}_layer1']

        for name in recv_names:
            ack(name)

        timeline['side_head0'] = time.time_ns()
        x_side = self.avgpool(x_side)
        x_side = torch.flatten(x_side, 1)
        x_side = self.upsample(x_side)
        x_side = self.head(x_side)
        timeline['side_head1'] = time.time_ns()
        timeline[f'side_end'] = timeline['side_head1']
        return x_side, timeline

        # === NEW: inside the model class (VGG/ResNet/ViT) ===








































# Functional Test
if __name__ == '__main__':

    # Define the model
    model = ResNet()
    print(model)

    # Get the model size
    print(model.get_model_size())
