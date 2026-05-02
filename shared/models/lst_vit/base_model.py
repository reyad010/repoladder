# Description: VGG16 model definition and forward pass
# from collections import OrderedDict
from modules.layers import *
import timm
from modules.mix import MixedEdge_NoParams
from utils.flops_computation import linear_flops, conv2d_flops
from utils.connectors import _match_dim_np
from utils.tensor_transfer_channel import send_value, recv_value
try: from shmio import ack, tensor_hash
except Exception as e: ack, tensor_hash = None, None
import torch.cuda.nvtx as nvtx

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

class ViT(nn.Module):
    def __init__(self, layers_dict, num_classes=100):
        super(ViT, self).__init__()
        self.fake_forward_feature = None
        self.model_name = 'vit-base'
        self.separate_device = False
        # Define Ladder Blocks
        self.connection_layers = nn.ModuleList(layers_dict['connection_layers'])

        # Define Side Blocks
        self.side_layers = nn.ModuleList(layers_dict['side_layers'])
        self.upsample = layers_dict['upsample']
        self.head = layers_dict['head']
        self.gate_params = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(13)])
        self.temperature = 0.1

        # Define Base ViT model. Lookup order for the pretrained weights:
        #   1. $VIT_WEIGHTS_PATH                                              (explicit override)
        #   2. <this file's dir>/vit_base_patch16_224.pth                     (artifact-bundled)
        #   3. ./models/vit_base_patch16_224.pth                              (original repo layout)
        #   4. timm pretrained download from HuggingFace (no network in SGX)
        # The pre-downloaded file is required for gpu-tee mode because the
        # SGX enclave has no network access.
        self.num_classes = num_classes
        import os
        _candidates = [
            os.environ.get('VIT_WEIGHTS_PATH'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vit_base_patch16_224.pth'),
            './models/vit_base_patch16_224.pth',
        ]
        _local_ckpt = next((p for p in _candidates if p and os.path.exists(p)), None)
        if _local_ckpt is not None:
            self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False, embed_dim=768)
            self.backbone.load_state_dict(torch.load(_local_ckpt))
        else:
            self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, embed_dim=768)
        self.backbone.head = None
        dim_backbone: int = self.backbone.embed_dim  # 768 for ViT‑base
        heads_backbone: int = self.backbone.blocks[0].attn.num_heads
        self.patch_embed = self.backbone.patch_embed
        self.cls_token = self.backbone.cls_token
        self.pos_embed = self.backbone.pos_embed
        self.pos_drop = self.backbone.pos_drop
        self.backbone_layers = nn.ModuleList(self.backbone.blocks)
        self.backbone_norm = self.backbone.norm
        self.dropout = nn.Dropout(0.1)

        self.transfer_type = 'shm'
        self._channels_last = False

    def forward(self, x=None, verbose=False, return_timeline=False, sync=False):
        def image2embed(images):
            B = images.size(0)
            x_back = self.patch_embed(images)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x_back = torch.cat((cls_tokens, x_back), dim=1)
            x_back = x_back + self.pos_embed[:, : x_back.size(1), :]
            x_back = self.pos_drop(x_back)
            x_back = self.dropout(x_back)
            return x_back
        chlast = getattr(self, "_channels_last", False)
        x = x if x is not None else self.get_dummy_inputs(device='cuda:0', channels_last=chlast)
        if verbose:print(f"x: {tensor_hash(x)[:5]}")
        timeline = {}
        nvtx.range_push("backbone forward")
        timeline["backbone_start"] = time.time_ns()
        # --- image -> embeddings (backbone input) ---
        x_backbone = image2embed(x)
        torch.cuda.synchronize()
        timeline["backbone_x_trans0"] = time.time_ns()
        x_cpu = x_backbone if not self.separate_device else x_backbone.to(self.side_device)
        torch.cuda.synchronize()
        timeline["backbone_x_trans1"] = time.time_ns()

        # --- backbone layers ---
        backbone_features = []
        for i, layer in enumerate(self.backbone_layers):
            timeline[f"backbone_f{i}_start"] = time.time_ns()
            x_backbone = layer(x_backbone)
            if sync: torch.cuda.synchronize()
            timeline[f"backbone_f{i}_end"] = time.time_ns()

            # Prepare features for side connections
            if isinstance(self.connection_layers[i], EmptyLayer):
                backbone_features.append(None)
                continue

            # Record async transfer with CUDA event
            if self.separate_device:
                timeline[f"backbone_f{i}_trans0"] = time.time_ns()
                event = torch.cuda.Event(blocking=False)
                x_backbone_cpu = x_backbone.to(self.side_device, non_blocking=True)
                event.record()  # mark copy completion
                timeline[f"backbone_f{i}_trans1"] = time.time_ns()
                backbone_features.append((x_backbone_cpu, event)) # event
            else:
                backbone_features.append((x_backbone, None))

        timeline["backbone_end"] = time.time_ns()

        # --- side path: input ladder ---

        timeline["side_start"] = time.time_ns()
        x_side = self.connection_layers[-1](x_cpu)
        timeline["side_input_ladder_end"] = time.time_ns()

        # --- side layers ---
        side_norm = None
        for i, layer in enumerate(self.side_layers):
            reduced_feature = None
            conn = self.connection_layers[i]
            key = f"f{i}"

            timeline[f"side_{key}_start"] = time.time_ns()

            # Wait for transfer event only when needed
            if (not isinstance(conn, EmptyLayer)) and backbone_features[i] is not None:
                feat, evt = backbone_features[i]
                if evt is not None:evt.synchronize()  # wait for its specific copy only
                timeline[f"side_{key}_con0"] = time.time_ns()
                reduced_feature = conn(feat)
                timeline[f"side_{key}_con1"] = time.time_ns()

            # Fuse reduced feature
            if reduced_feature is not None:
                timeline[f"side_{key}_add0"] = time.time_ns()
                # reduced_feature = self._apply_connector(reduced_feature, x_side, getattr(self, "layer_con_type", "pool"))
                ai = self.gate_params[i]
                # ui = torch.sigmoid(ai / self.temperature)
                # x_side = ui * x_side + (1 - ui) * reduced_feature
                ui = torch.sigmoid(ai / self.temperature)
                x_side = torch.lerp(x_side, reduced_feature, 1.0 - ui)
                timeline[f"side_{key}_add1"] = time.time_ns()

            # Side layer compute
            timeline[f"side_{key}_side0"] = time.time_ns()
            # x_side = self._coerce_x_to_layer_input(x_side, layer)
            x_side = layer(x_side)
            timeline[f"side_{key}_side1"] = time.time_ns()

            # Track normalization layer
            if isinstance(layer, ObfuscationBlock):
                cur_layer = layer.conv.active_op
                side_norm = getattr(cur_layer, "side_norm", side_norm)
            else:
                side_norm = getattr(layer, "side_norm", side_norm)

        # --- head ---
        x_cls = side_norm(x_side[:, 0, :])
        x_cls = self.dropout(x_cls)
        x_cls = self.upsample(x_cls)
        x_out = self.head(x_cls)
        timeline["side_end"] = time.time_ns()
        torch.cuda.synchronize()
        if verbose:print(f"x_out: {tensor_hash(x_out)[:5]}")
        if return_timeline:return x_out, timeline
        return x_out

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
        return

    # Initialize all weights = backbone weights + obfuscation block weights
    def init_model(self, state_dict):
        # already initialized in init
       return

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
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                module.eval()

    def freeze_backbone(self):
        self.freeze_params()

    # ------------------------------------------------------------------
    #  Analytical static info (no inference, non-destructive)
    # ------------------------------------------------------------------
    @staticmethod
    def _vit_backbone_finetune_flops(B: int = 100, N: int = 197, D: int = 768,
                                     H: int = 12, head_dim: int = 64,
                                     num_classes: int = 100,
                                     mlp_hidden: int = 3072) -> int:
        """Per-batch FLOPs for a full ViT-Base forward (frozen-backbone baseline)."""
        flops = 0
        # patch embed: Conv2d(3, 768, k=16, s=16) -> (B, 14, 14, 768)
        flops += B * conv2d_flops(3, D, 16, 16, 14, 14)
        for _ in range(12):
            # attn: fused qkv (768 -> 2304) + proj (768 -> 768)
            flops += linear_flops(D, 3 * D, B * N)
            flops += linear_flops(D, D, B * N)
            # attn matmul: QK^T and AV (treated symmetrically)
            flops += 2 * B * H * N * head_dim * N  # QK^T
            flops += 2 * B * H * N * N * head_dim  # AV
            # mlp
            flops += linear_flops(D, mlp_hidden, B * N)
            flops += linear_flops(mlp_hidden, D, B * N)
        # cls head: (768 -> num_classes)
        flops += linear_flops(D, num_classes, B)
        return int(flops)

    def _vit_lst_side_block_flops(self, side_layer, B: int = 100, N: int = 197) -> int:
        """FLOPs for one ViT side block analytically (no forward needed)."""
        if isinstance(side_layer, EmptyLayer):
            return 0
        if isinstance(side_layer, ViTBlock):
            # The ViTBlock.get_flops uses input_shape and 4*linear_flops(D,D) + MLP.
            # Compute analytically from rf/embed_dim attributes.
            # dim_side = embed_dim // rf
            # MLP hidden = dim_side * mlp_ratio (default mlp_ratio=4.0 -> 4*dim_side)
            # The first Linear's out_features is the hidden dim.
            try:
                D = side_layer.norm1.normalized_shape[0]
            except Exception:
                # Fallback: guess from rf
                rf = getattr(side_layer, 'reduction_factor', 1)
                D = 768 // max(rf, 1)
            hidden = side_layer.mlp[0].out_features if hasattr(side_layer, 'mlp') else (D * 4)
            attn_flops = 4 * linear_flops(D, D, B * N)  # QKV proj fused factor + out proj
            mlp_flops = linear_flops(D, hidden, B * N) + linear_flops(hidden, D, B * N)
            return int(attn_flops + mlp_flops)
        # Other side-layer types (rare for ViT) — best effort
        if hasattr(side_layer, 'get_flops') and getattr(side_layer, 'input_shape', None) is not None:
            try:
                f, _ = side_layer.get_flops()
                return int(f)
            except Exception:
                return 0
        return 0

    def _vit_connection_flops(self, conn_layer, B: int = 100, N: int = 197, D_in: int = 768) -> int:
        """FLOPs for one connection layer (Linear-style for ViT)."""
        if conn_layer is None or isinstance(conn_layer, EmptyLayer):
            return 0
        if isinstance(conn_layer, ReducedLinearLayer):
            return int(linear_flops(conn_layer.in_channels, conn_layer.out_channels, B * N))
        if isinstance(conn_layer, LALinear):
            return int(linear_flops(conn_layer.down.in_features, conn_layer.rank, B * N) +
                       linear_flops(conn_layer.rank, conn_layer.up.out_features, B * N))
        if isinstance(conn_layer, nn.Linear):
            return int(linear_flops(conn_layer.in_features, conn_layer.out_features, B * N))
        return 0

    def _build_lst_transfer_shapes_static(self, B: int = 100, N: int = 197, D: int = 768):
        """Mirror convert_to_backbone_model's shape construction (LST: GPU->TEE only)."""
        shapes = {'x': (B, N, D)}
        if hasattr(self, 'connection_layers'):
            n_main = len(self.side_layers) if hasattr(self, 'side_layers') else 12
            for i, conn in enumerate(self.connection_layers):
                if i >= n_main:
                    continue
                if not isinstance(conn, EmptyLayer):
                    shapes[f"f{i}"] = (B, N, D)
        return shapes

    def get_static_info(self):
        """Analytical static info for LST-ViT — pure (no forward, non-destructive).

        Notes:
          - LST is unidirectional (GPU->TEE only); both online_mask_flops and
            offline_mask_flops are 0 by definition (no mask add/sub on transferred tensors).
        """
        # Save and freeze for trainable param count, restore after
        prev_state = [(p, p.requires_grad) for p in self.parameters()]
        try:
            self.freeze_backbone()
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        finally:
            for p, rg in prev_state:
                p.requires_grad = rg

        trainable_pct = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0
        model_params_M = total_params / 1e6

        B, N, D = 100, 197, 768
        num_classes = self.num_classes

        # ---- backbone_finetune_flops: full ViT-Base forward ----
        backbone_flops = self._vit_backbone_finetune_flops(
            B=B, N=N, D=D, H=12, head_dim=64, num_classes=num_classes, mlp_hidden=4 * D
        )

        # ---- online_adapter_flops: input ladder + per-stage (active conn + side) + upsample + head ----
        adapter_flops = 0

        n_main = len(self.side_layers) if hasattr(self, 'side_layers') else 12

        # Input ladder = connection_layers[-1] applied to (B, N, D=768)
        if hasattr(self, 'connection_layers') and len(self.connection_layers) > n_main:
            input_ladder = self.connection_layers[-1]
            adapter_flops += self._vit_connection_flops(input_ladder, B=B, N=N, D_in=D)

        # Per-stage: active connection layer + side layer
        for i in range(n_main):
            conn = self.connection_layers[i] if hasattr(self, 'connection_layers') else None
            side = self.side_layers[i] if hasattr(self, 'side_layers') else None
            if conn is not None and not isinstance(conn, EmptyLayer):
                adapter_flops += self._vit_connection_flops(conn, B=B, N=N, D_in=D)
            if side is not None:
                adapter_flops += self._vit_lst_side_block_flops(side, B=B, N=N)

        # Upsample: ReducedLinearLayer or Linear (applied to CLS token only -> instances=B)
        upsample = getattr(self, 'upsample', None)
        if isinstance(upsample, ObfuscationBlock):
            upsample = upsample.conv if hasattr(upsample, 'conv') else upsample
            if isinstance(upsample, MixedEdge_NoParams):
                upsample = upsample.active_op
        if isinstance(upsample, ReducedLinearLayer):
            adapter_flops += int(linear_flops(upsample.in_channels, upsample.out_channels, B))
        elif isinstance(upsample, nn.Linear):
            adapter_flops += int(linear_flops(upsample.in_features, upsample.out_features, B))

        # Head: nn.Linear(768, num_classes)
        head = getattr(self, 'head', None)
        if isinstance(head, nn.Linear):
            adapter_flops += int(linear_flops(head.in_features, head.out_features, B))

        adapter_pct = (adapter_flops / backbone_flops * 100.0) if backbone_flops > 0 else 0.0

        # Transfer shapes (LST: GPU->TEE only)
        transfer_shapes = self._build_lst_transfer_shapes_static(B=B, N=N, D=D)
        feature_transfer_count = len(transfer_shapes)
        feature_transfer_bytes = 0
        for s in transfer_shapes.values():
            n_elem = 1
            for d in s:
                n_elem *= int(d)
            feature_transfer_bytes += n_elem * 4  # float32

        return {
            "model_name": "vit-base",
            "peft": "lst",
            "trainable_params": int(trainable_params),
            "total_params": int(total_params),
            "trainable_pct": float(trainable_pct),
            "model_params_M": float(model_params_M),
            "online_adapter_flops": int(adapter_flops),
            "backbone_finetune_flops": int(backbone_flops),
            "online_adapter_flops_pct": float(adapter_pct),
            "online_mask_flops": 0,
            "offline_mask_flops": 0,
            "feature_transfer_bytes": int(feature_transfer_bytes),
            "feature_transfer_count": int(feature_transfer_count),
        }

    def get_dummy_inputs(self, device='cpu', channels_last=False):
        if channels_last:
            dummy_inputs = (torch.randn(100, 3, 224, 224).contiguous(memory_format=torch.channels_last).to(device))
        else:
            dummy_inputs = (torch.randn(100, 3, 224, 224).to(device))
        return dummy_inputs

    def get_dummy_input_feature(self, device='cpu', channels_last=False):

        dummy_inputs = (torch.randn(100, 197, 768).to(device))
        return dummy_inputs

    def get_dummy_features(self, device='cpu', channels_last=False):
        dummy_features = [
            torch.rand((100, 197, 768)).to(device),
            torch.rand((100, 197, 768)).to(device),
            torch.rand((100, 197, 768)).to(device),
            torch.rand((100, 197, 768)).to(device),
            torch.rand((100, 197, 768)).to(device),
            torch.rand((100, 197, 768)).to(device),
            torch.rand((100, 197, 768)).to(device),
            torch.rand((100, 197, 768)).to(device),
            torch.rand((100, 197, 768)).to(device),
            torch.rand((100, 197, 768)).to(device),
            torch.rand((100, 197, 768)).to(device),
            torch.rand((100, 197, 768)).to(device),
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

    def convert_to_backbone_model(self, device):
        # build flags indicating if we need current layer's fature map
        self.output_feature_flag = []
        for connection_layer in self.connection_layers:
            self.output_feature_flag.append(not isinstance(connection_layer, EmptyLayer))

        # get transfer tensor's name and shape
        shapes = {'x': tuple(self.get_dummy_input_feature(device='cpu').shape)}
        feats = self.get_dummy_features(device='cpu', channels_last=True) or []
        num_feat = len(feats)  # backbone-aligned features only (no input-ladder)

        for i, conn in enumerate(self.connection_layers):
            # Skip input-ladder (and any extra trailing connections beyond feats)
            if i >= num_feat: continue
            # Only add a slot if this connection is active and the feature exists
            if (not isinstance(conn, EmptyLayer)) and feats[i] is not None:
                shapes[f"f{i}"] = tuple(feats[i].shape)
        self.transfer_shapes = shapes

        del self.side_layers, self.connection_layers, self.gate_params, self.upsample, self.head
        self._channels_last = False
        self.forward = self.backbone_forward_eager_mode
        for layer in self.backbone_layers:
            layer.to(device)

    def convert_to_side_model(self, device):
        del self.backbone_layers
        self.side_device_init(device)

        # predefine transfer shapes so recv_value(type='tensor') can reshape
        shapes = {'x': tuple(self.get_dummy_input_feature(device='cpu').shape)}

        feats = self.get_dummy_features(device='cpu', channels_last=True) or []
        num_feat = len(feats)  # backbone-aligned features only (no input-ladder)

        for i, conn in enumerate(self.connection_layers):
            # Skip input-ladder (and any extra trailing connections beyond feats)
            if i >= num_feat: continue
            # Only add a slot if this connection is active and the feature exists
            if (not isinstance(conn, EmptyLayer)) and feats[i] is not None:
                shapes[f"f{i}"] = tuple(feats[i].shape)

        self.transfer_shapes = shapes
        self._channels_last = False
        self.forward = self.side_forward_eager_mode

    def backbone_forward_eager_mode(self, shm_tensors, iter, verbose=False, inputs=None):
        time_line = {}
        check_value = '0' if iter == 11 else '0'
        chlast = getattr(self, "_channels_last", True)
        x = inputs if inputs is not None else self.get_dummy_inputs(device='cuda:0', channels_last=chlast)
        # if verbose: print(f'x: {tensor_hash(x)[:5]}')
        nvtx.range_push(f"backbone forward {iter}")
        time_line['backbone_start'] = time.time_ns()
        def image2embed(images):
            B = images.size(0)
            x_back = self.patch_embed(images)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x_back = torch.cat((cls_tokens, x_back), dim=1)
            x_back = x_back + self.pos_embed[:, : x_back.size(1), :]
            x_back = self.pos_drop(x_back)
            x_back = self.dropout(x_back)
            return x_back
        x = image2embed(x)
        torch.cuda.synchronize()
        time_line['backbone_x_trans0'] = time.time_ns()
        send_names = [f'x']
        send_value(f'x', x, 'tensor', method=self.transfer_type, shared_memory=shm_tensors['x'], channels_last=chlast, check_value=check_value)
        time_line['backbone_x_trans1'] = time.time_ns()

        # backbone forward
        for i, layer in enumerate(self.backbone_layers):
            key = f"f{i}"
            time_line[f'backbone_{key}_start'] = time.time_ns()
            time_line[f'backbone_{key}_layer0'] = time.time_ns()
            x = layer(x)
            torch.cuda.synchronize()
            time_line[f'backbone_{key}_layer1'] = time.time_ns()

            if hasattr(self, "output_feature_flag") and self.output_feature_flag[i]:
                if self.transfer_type == 'osp':
                    feat_cpu = x.to('cpu', non_blocking=True)
                    send_value(key, feat_cpu, 'tensor', method=self.transfer_type, channels_last=chlast, check_value=check_value)
                elif self.transfer_type == 'shm':
                    # when using shm method, the tensor is directly transferred to shared memory (single copy)
                    time_line[f'backbone_{key}_trans0'] = time.time_ns()
                    send_value(key, x, 'tensor', method=self.transfer_type, shared_memory=shm_tensors[key], channels_last=chlast, check_value=check_value)
                    time_line[f'backbone_{key}_trans1'] = time.time_ns()
                send_names.append(key)
            time_line[f'backbone_{key}_end'] = time.time_ns()

        torch.cuda.synchronize()
        time_line[f'backbone_end'] = time.time_ns()
        nvtx.range_pop()
        return time_line

    def side_forward_eager_mode(self, iter, verbose=False):
        time_line = {}
        chlast = getattr(self, "_channels_last", False)
        shapes = self.transfer_shapes
        check_value = '0' if iter in [11] else '0'

        time_line['side_x_recv0'] = time.time_ns()
        x_side_in = recv_value(f'x', 'tensor', shape=tuple(shapes['x']), method=self.transfer_type, channels_last=chlast, check_value=check_value)
        time_line['side_x_recv1'] = time.time_ns()

        time_line['side_start'] = time.time_ns()
        time_line[f'side_input_ladder_start'] = time_line['side_start']
        x_side = self.connection_layers[-1](x_side_in)
        time_line[f'side_input_ladder_end'] = time.time_ns()
        recv_names = [f'x']

        side_norm = None
        for i, layer in enumerate(self.side_layers):
            conn = self.connection_layers[i]
            key = f"f{i}"
            layer_name = f'f{i}'
            time_line[f'side_{key}_start'] = time.time_ns()
            if not isinstance(conn, EmptyLayer) and (layer_name in shapes):
                time_line[f'side_{key}_recv0'] = time.time_ns()
                feat_i = recv_value(key, 'tensor', shape=tuple(shapes[layer_name]),method=self.transfer_type, channels_last=chlast, check_value=check_value)
                time_line[f'side_{key}_recv1'] = time.time_ns()
                time_line[f'side_{key}_con0'] = time.time_ns()
                reduced_feature = conn(feat_i)
                time_line[f'side_{key}_con1'] = time.time_ns()
                recv_names.append(key)

                # Addition: connection feature + side feature
                time_line[f'side_{key}_add0'] = time.time_ns()
                # reduced_feature = self._apply_connector(reduced_feature, x_side, getattr(self, 'layer_con_type', 'pool'))
                ai = self.gate_params[i]
                # ui = torch.sigmoid(ai / self.temperature)
                # x_side = ui * x_side + (1 - ui) * reduced_feature
                ui = torch.sigmoid(ai / self.temperature)
                x_side = torch.lerp(x_side, reduced_feature, 1.0 - ui)
                time_line[f'side_{key}_add1']  = time.time_ns()

            time_line[f'side_{key}_layer0'] = time.time_ns()
            # x_side = self._coerce_x_to_layer_input(x_side, layer)
            x_side = layer(x_side)
            time_line[f'side_{key}_layer1'] = time.time_ns()

            if isinstance(layer, ObfuscationBlock):  # used in super_net
                # cur_layer = layer.conv.candidate_ops[layer.conv.active_index[0]]
                cur_layer = layer.conv.active_op
                side_norm = cur_layer.side_norm if hasattr(cur_layer, 'side_norm') else side_norm
                # print(f'enter obfuscation block')
                # print(side_norm)
            else:  # used in converted model
                side_norm = layer.side_norm if hasattr(layer, 'side_norm') else side_norm

            time_line[f'side_{key}_end'] = time.time_ns()

        for name in recv_names:
            ack(name)
        # x_cls = self.side_norm(x_side[:, 0, :])
        time_line['side_head0'] = time.time_ns()
        x_cls = side_norm(x_side[:, 0, :])
        x_cls = self.dropout(x_cls)
        x_cls = self.upsample(x_cls)
        x_cls = self.head(x_cls)
        time_line['side_head1'] = time.time_ns()
        time_line[f'side_end'] = time_line['side_head1']
        # if verbose: print(f'x_out: {tensor_hash(x_cls)[:5]}')
        return x_cls, time_line



# Functional Test
if __name__ == '__main__':

    # Define the model
    model = ViT({'connection_layers': None, 'side_layers': None, 'upsample': None, 'head': None})
    print(model)

    # # Get the model size
    # print(model.get_model_size())

