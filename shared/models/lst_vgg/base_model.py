# Description: VGG16 model definition and forward pass
import torch
import time
from modules.layers import *
from utils.connectors import _match_channels_np, _match_spatial_np
from utils.tensor_transfer_channel import send_value, recv_value, send_value_no_transfer, recv_value_no_transfer
from utils.flops_computation import conv2d_flops, linear_flops
try: from shmio import ack, tensor_hash
except Exception as e: ack, tensor_hash = None, None
import torch.cuda.nvtx as nvtx
# import torchvision
import collections

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

class VGGNet(nn.Module):
    def __init__(self, layers_dict, num_classes=10):
        super(VGGNet, self).__init__()
        self.layer_con_type = 'pool'
        self.fake_forward_feature = None
        oc = [64, 128, 256, 512, 4096] # original channels/features
        self.oc = oc
        self.model_name = 'vgg'
        self.separate_device = False
        # Define Ladder Blocks
        self.connection_layers = nn.ModuleList(layers_dict['connection_layers'])

        # Define Side Blocks
        self.side_layers = nn.ModuleList(layers_dict['side_layers'])
        self.upsample = layers_dict['upsample']
        self.head = layers_dict['head']
        self.gate_params = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(14)])
        self.temperature = 0.1


        # Define Base VGG16 model
        backbone_config = [
            {
                "type": "Conv2d",
                "in_channels": 3,
                "out_channels": 64,
                "kernel_size": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "padding": [
                    1,
                    1
                ],
                "bias": True
            },
            {
                "type": "Conv2d",
                "in_channels": 64,
                "out_channels": 64,
                "kernel_size": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "padding": [
                    1,
                    1
                ],
                "bias": True
            },
            {
                "type": "Conv2d",
                "in_channels": 64,
                "out_channels": 128,
                "kernel_size": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "padding": [
                    1,
                    1
                ],
                "bias": True
            },
            {
                "type": "Conv2d",
                "in_channels": 128,
                "out_channels": 128,
                "kernel_size": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "padding": [
                    1,
                    1
                ],
                "bias": True
            },
            {
                "type": "Conv2d",
                "in_channels": 128,
                "out_channels": 256,
                "kernel_size": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "padding": [
                    1,
                    1
                ],
                "bias": True
            },
            {
                "type": "Conv2d",
                "in_channels": 256,
                "out_channels": 256,
                "kernel_size": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "padding": [
                    1,
                    1
                ],
                "bias": True
            },
            {
                "type": "Conv2d",
                "in_channels": 256,
                "out_channels": 256,
                "kernel_size": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "padding": [
                    1,
                    1
                ],
                "bias": True
            },
            {
                "type": "Conv2d",
                "in_channels": 256,
                "out_channels": 512,
                "kernel_size": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "padding": [
                    1,
                    1
                ],
                "bias": True
            },
            {
                "type": "Conv2d",
                "in_channels": 512,
                "out_channels": 512,
                "kernel_size": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "padding": [
                    1,
                    1
                ],
                "bias": True
            },
            {
                "type": "Conv2d",
                "in_channels": 512,
                "out_channels": 512,
                "kernel_size": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "padding": [
                    1,
                    1
                ],
                "bias": True
            },
            {
                "type": "Conv2d",
                "in_channels": 512,
                "out_channels": 512,
                "kernel_size": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "padding": [
                    1,
                    1
                ],
                "bias": True
            },
            {
                "type": "Conv2d",
                "in_channels": 512,
                "out_channels": 512,
                "kernel_size": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "padding": [
                    1,
                    1
                ],
                "bias": True
            },
            {
                "type": "Conv2d",
                "in_channels": 512,
                "out_channels": 512,
                "kernel_size": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "padding": [
                    1,
                    1
                ],
                "bias": True
            },
        ]
        self.num_classes = num_classes

        self.relu = nn.ReLU(inplace=False)
        self.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_side = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(p=0.5)

        self.backbone_layers = nn.ModuleList()
        for cfg in backbone_config:
            self.backbone_layers.append(nn.Conv2d(cfg['in_channels'], cfg['out_channels'], kernel_size=cfg['kernel_size'],
                                                  padding=cfg['padding'], stride=cfg['stride'], bias=cfg['bias']))

        # define communication method
        self.transfer_type = 'shm'
        self._channels_last = True
        # self.init_weight()
        # print('current state dict')
        # for k, v in self.state_dict().items():
        #     print(k, v.shape)
        # exit()

    @torch.no_grad()
    def init_weight(self, *, strict: bool = True):
        # torchvision API compatibility (new + old)

        try:
            import torchvision
            from torchvision.models import VGG16_Weights
            tv_model = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except Exception:
            tv_model = torchvision.models.vgg16(pretrained=True)

        tv_sd = tv_model.state_dict()
        cur_sd = self.state_dict()
        converted = collections.OrderedDict()

        tv_conv_idx = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        key_map = {f"features.{t}": f"backbone_layers.{i}" for i, t in enumerate(tv_conv_idx)}

        for k, v in tv_sd.items():
            new_k = k

            # remap conv layers from torchvision -> our backbone_layers
            for src, dst in key_map.items():
                if k.startswith(src + "."):
                    rest = k[len(src) + 1:]  # "weight" or "bias"
                    new_k = dst + "." + rest
                    break

            if new_k in cur_sd:
                if tuple(cur_sd[new_k].shape) != tuple(v.shape):
                    print(
                        f"skip initialize layer: {new_k}, "
                        f"torchvision shape: {tuple(v.shape)}, "
                        f"cur model shape: {tuple(cur_sd[new_k].shape)}"
                    )
                    continue

                ref = cur_sd[new_k]
                converted[new_k] = v.to(dtype=ref.dtype, device=ref.device).contiguous()

        cur_sd.update(converted)
        self.load_state_dict(cur_sd, strict=strict)

        print(f"VGGLST/VGGNet backbone initialized from torchvision VGG16 ({len(converted)} tensors loaded)")
        return converted

    def forward(self, x=None, verbose=False, return_timeline=False, sync=False):
        """
        VGG forward with timeline + safe side-path transfer (copy happens in side loop).
        If return_timeline=True -> returns (y, timeline).
        If sync=True -> cuda synchronize at key points (slower but precise).
        """
        chlast = self._channels_last
        x = x if x is not None else self.get_dummy_inputs(device='cuda:0', channels_last=chlast)
        x_side_in = x.to('cpu', non_blocking=True) if self.separate_device else x
        torch.cuda.synchronize()
        timeline = {}

        # ===== Backbone =====
        # nvtx.range_push("backbone forward")
        if return_timeline: timeline["backbone_start"] = time.time_ns()
        x_backbone = x
        backbone_features = []  # keep features on backbone device; do not transfer here

        for i, layer in enumerate(self.backbone_layers):
            key = f"f{i}"
            if return_timeline: timeline[f"backbone_{key}_start"] = time.time_ns()
            x_backbone = layer(x_backbone)
            if sync:torch.cuda.synchronize()
            if return_timeline: timeline[f"backbone_{key}_end"] = time.time_ns()

            # Prepare features for side connections
            if isinstance(self.connection_layers[i], EmptyLayer):
                backbone_features.append(None)
            else:
                # Record async transfer with CUDA event
                if self.separate_device:
                    if return_timeline: timeline[f"backbone_f{i}_trans0"] = time.time_ns()
                    event = torch.cuda.Event(blocking=False)
                    x_backbone_cpu = x_backbone.to(self.side_device, non_blocking=True)
                    event.record()  # mark copy completion
                    if return_timeline: timeline[f"backbone_f{i}_trans1"] = time.time_ns()
                    backbone_features.append((x_backbone_cpu, event))  # event
                else:
                    backbone_features.append((x_backbone, None))

            # backbone activations/pools
            x_backbone = self.relu(x_backbone)
            if i in [1, 3, 6, 9, 12]:
                x_backbone = self.MaxPool2d(x_backbone)

        if return_timeline:timeline["backbone_end"] = time.time_ns()
        # nvtx.range_pop()

        # ===== Side path =====
        if return_timeline:timeline["side_start"] = time.time_ns()

        # input ladder
        x_side = self.connection_layers[-1](x_side_in)
        if return_timeline:timeline["side_input_ladder_end"] = time.time_ns()

        # side layers
        for i, layer in enumerate(self.side_layers):
            key = f"f{i}"
            conn = self.connection_layers[i]

            if return_timeline:timeline[f"side_{key}_start"] = time.time_ns()

            # Transfer the paired backbone feature **here** (if active), then wait for its event
            if not isinstance(conn, EmptyLayer):
                feat, evt = backbone_features[i]
                if evt is not None: evt.synchronize()  # wait for its specific copy only
                if return_timeline:timeline[f"side_{key}_con0"] = time.time_ns()
                reduced_feature = conn(feat)
                if return_timeline:timeline[f"side_{key}_con1"] = time.time_ns()

                # Addition reduced feature
                if return_timeline:timeline[f"side_{key}_add0"] = time.time_ns()
                # reduced_feature = self._apply_connector(reduced_feature, x_side, getattr(self, "layer_con_type", "pool"), is_vit=False)
                # ui = torch.sigmoid(self.gate_params[i] / self.temperature)
                # x_side = ui * x_side + (1 - ui) * reduced_feature
                ui = torch.sigmoid(self.gate_params[i] / self.temperature)
                x_side = torch.lerp(x_side, reduced_feature, 1.0 - ui)
                if return_timeline:timeline[f"side_{key}_add1"] = time.time_ns()

            # side layer compute (+ optional pool after specific layers)
            if return_timeline:timeline[f"side_{key}_layer0"] = time.time_ns()
            # x_side = self._coerce_x_to_layer_input(x_side, layer)
            x_side = layer(x_side)
            if return_timeline:timeline[f"side_{key}_layer1"] = time.time_ns()

            if i in [1, 3, 6, 9, 12]:x_side = self.MaxPool2d(x_side)

            if sync:torch.cuda.synchronize()
            if return_timeline:timeline[f"side_{key}_end"] = time.time_ns()

        # head
        if return_timeline: timeline['side_head0'] = time.time_ns()
        x_side = self.avgpool_side(x_side)
        x_side = torch.flatten(x_side, 1)
        # x_side = self.dropout(x_side)
        x_side = self.upsample(x_side)
        # x_side = self.relu(x_side)
        # x_side = self.dropout(x_side)
        y = self.head(x_side)
        if return_timeline: timeline['side_head1'] = time.time_ns()

        if sync:torch.cuda.synchronize()
        if return_timeline:timeline["side_end"] = time.time_ns()

        if return_timeline:
            return y, timeline
        return y

    def side_device_init(self, side_device, blocking=False):

        # self.blocking = False
        self.side_device = side_device
        for layer in self.side_layers:
            layer.to(side_device)
        for layer in self.connection_layers:
            layer.to(side_device)
        self.upsample.to(side_device)
        self.head.to(side_device)
        self.gate_params.to(side_device)

        self.relu_side = nn.ReLU(inplace=False).to(side_device)
        self.MaxPool2d_side = nn.MaxPool2d(kernel_size=2, stride=2).to(side_device)
        self.avgpool_side = self.avgpool_side.to(side_device)
        # self.dropout_side = nn.Dropout(p=0.5).to(side_device)
        self.separate_device = True

    # Map trained weights to backbone model
    def map_weights(self, state_dict):
        key_in_state_dict = ['features.0.weight', 'features.0.bias', 'features.2.weight', 'features.2.bias', 'features.5.weight', 'features.5.bias', 'features.7.weight', 'features.7.bias', 'features.10.weight', 'features.10.bias', 'features.12.weight', 'features.12.bias', 'features.14.weight', 'features.14.bias', 'features.17.weight', 'features.17.bias', 'features.19.weight', 'features.19.bias', 'features.21.weight', 'features.21.bias', 'features.24.weight', 'features.24.bias', 'features.26.weight', 'features.26.bias', 'features.28.weight', 'features.28.bias',]
        # key_in_cur_model = ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'conv3.weight', 'conv3.bias', 'conv4.weight', 'conv4.bias', 'conv5.weight', 'conv5.bias', 'conv6.weight', 'conv6.bias', 'conv7.weight', 'conv7.bias', 'conv8.weight', 'conv8.bias', 'conv9.weight', 'conv9.bias', 'conv10.weight', 'conv10.bias', 'conv11.weight', 'conv11.bias', 'conv12.weight', 'conv12.bias', 'conv13.weight', 'conv13.bias']
        key_in_cur_model = []
        for i in range(13):
            key_in_cur_model.append(f'backbone_layers.{i}.weight')
            key_in_cur_model.append(f'backbone_layers.{i}.bias')

        new_state_dict = OrderedDict()
        for k1, k2 in zip(key_in_state_dict, key_in_cur_model):
            new_state_dict[k2] = state_dict[k1]

        self.load_state_dict(new_state_dict, strict=False)
        print('vgg model initialized')

    # Initialize all weights = backbone weights + obfuscation block weights
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
                # print('Freezing:', name)
                if param.requires_grad:
                    param.requires_grad = False

        # Freeze the BN running mean and variance
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def freeze_backbone(self):
        self.freeze_params()

    # ----------------------------
    # Static analytical info
    # ----------------------------
    @staticmethod
    def _vgg_backbone_shapes():
        """Return per-conv (Cin, Cout, kH, kW, Hout, Wout) for the 13 backbone convs at B=100, 64x64.
        Pools (stride 2) happen after conv indices {1, 3, 6, 9, 12}.
        """
        cfgs = [
            (3,   64,  64, 64),
            (64,  64,  64, 64),
            (64,  128, 32, 32),
            (128, 128, 32, 32),
            (128, 256, 16, 16),
            (256, 256, 16, 16),
            (256, 256, 16, 16),
            (256, 512, 8, 8),
            (512, 512, 8, 8),
            (512, 512, 8, 8),
            (512, 512, 4, 4),
            (512, 512, 4, 4),
            (512, 512, 4, 4),
        ]
        return [(cin, cout, 3, 3, h, w) for (cin, cout, h, w) in cfgs]

    @staticmethod
    def _vgg_feature_shapes(B=100):
        """Per-stage backbone feature map shape after conv i (pre relu/pool)."""
        shapes_chw = [
            (64, 64, 64),
            (64, 64, 64),
            (128, 32, 32),
            (128, 32, 32),
            (256, 16, 16),
            (256, 16, 16),
            (256, 16, 16),
            (512, 8, 8),
            (512, 8, 8),
            (512, 8, 8),
            (512, 4, 4),
            (512, 4, 4),
            (512, 4, 4),
        ]
        return [(B, c, h, w) for (c, h, w) in shapes_chw]

    def _build_lst_transfer_shapes(self):
        """Mirror convert_to_backbone_model's shape-construction logic without modifying state.
        Only active connection_layers (non-EmptyLayer) for backbone-aligned features get a slot.
        Always include 'x' for the initial input. All entries are GPU->TEE.
        """
        x_shape = (100, 3, 64, 64)
        shapes = {'x': x_shape}
        feat_shapes = self._vgg_feature_shapes(B=100)
        if hasattr(self, 'connection_layers'):
            num_feat = len(feat_shapes)
            for i, conn in enumerate(self.connection_layers):
                if i >= num_feat:
                    continue
                if not isinstance(conn, EmptyLayer):
                    shapes[f"f{i}"] = feat_shapes[i]
        return shapes

    def get_static_info(self):
        """Analytical static info — pure (no inference, restores requires_grad)."""
        # Save and freeze for trainable param count
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

        B = 100
        backbone_shapes = self._vgg_backbone_shapes()
        feat_shapes = self._vgg_feature_shapes(B=B)

        # ---- backbone_finetune_flops: 13 convs + linear head ----
        backbone_flops = 0
        for (cin, cout, kh, kw, hout, wout) in backbone_shapes:
            backbone_flops += B * conv2d_flops(cin, cout, kh, kw, hout, wout)
        # VGG has avgpool(7,7) + a Linear head 512 -> num_classes (per VGGNet: self.head)
        # backbone here only counts: 13 convs (avgpool free) + LST head Linear is part of side.
        # The "frozen backbone full forward" — for the LST baseline we still include the
        # original VGG classifier baseline. To be consistent with the "% baseline" definition,
        # treat backbone_finetune_flops as: 13 conv FLOPs only (avgpool free; backbone has
        # no separate trainable head in this codebase: the head lives on the side path).

        # ---- online_adapter_flops: side network (connection + side + upsample + head) ----
        adapter_flops = 0

        # 13 backbone-aligned connection layers + side layers
        if hasattr(self, 'connection_layers') and hasattr(self, 'side_layers'):
            num_main = len(self.side_layers)
            for i in range(num_main):
                conn = self.connection_layers[i]
                side = self.side_layers[i]

                # connection layer: input shape = backbone feature shape at stage i
                cin_b, cout_b, _, _, h_b, w_b = backbone_shapes[i]
                if isinstance(conn, ReducedConvLayer):
                    in_c = conn.in_channels
                    out_c = conn.out_channels
                    kh, kw = (conn.kernel_size, conn.kernel_size) if isinstance(conn.kernel_size, int) else conn.kernel_size
                    # 1x1 connection: spatial preserved
                    adapter_flops += B * conv2d_flops(in_c, out_c, kh, kw, h_b, w_b, conn.groups)

                # side layer: input shape uses the side stream channels.
                # For VGG the side layer at position i has shape matching feat_shape[i] but with reduced channels.
                # At side layer i, input spatial == feat_shapes[i][2:4] (post pool from i-1 already happened on side stream).
                # Per side cfg in ladder_side_tune_model: side input ch = B_in_channels[1:] + B_out_channels[-1:]
                if isinstance(side, ReducedConvLayer):
                    in_c = side.in_channels
                    out_c = side.out_channels
                    kh, kw = (side.kernel_size, side.kernel_size) if isinstance(side.kernel_size, int) else side.kernel_size
                    h_s, w_s = h_b, w_b
                    adapter_flops += B * conv2d_flops(in_c, out_c, kh, kw, h_s, w_s, side.groups)

            # input ladder = connection_layers[-1] applied to (B,3,64,64)
            if len(self.connection_layers) > num_main:
                inp_conn = self.connection_layers[-1]
                if isinstance(inp_conn, ReducedConvLayer):
                    in_c = inp_conn.in_channels
                    out_c = inp_conn.out_channels
                    kh, kw = (inp_conn.kernel_size, inp_conn.kernel_size) if isinstance(inp_conn.kernel_size, int) else inp_conn.kernel_size
                    adapter_flops += B * conv2d_flops(in_c, out_c, kh, kw, 64, 64, inp_conn.groups)

        # upsample: ReducedLinearLayer (in -> out)
        upsample = getattr(self, 'upsample', None)
        if isinstance(upsample, ReducedLinearLayer):
            adapter_flops += linear_flops(upsample.in_channels, upsample.out_channels, B)
        elif isinstance(upsample, nn.Linear):
            adapter_flops += linear_flops(upsample.in_features, upsample.out_features, B)

        # head: nn.Linear(512, num_classes)
        head = getattr(self, 'head', None)
        if isinstance(head, nn.Linear):
            adapter_flops += linear_flops(head.in_features, head.out_features, B)

        adapter_pct = (adapter_flops / backbone_flops * 100.0) if backbone_flops > 0 else 0.0

        # transfer shapes (LST: GPU->TEE only)
        transfer_shapes = self._build_lst_transfer_shapes()
        feature_transfer_count = len(transfer_shapes)
        feature_transfer_bytes = sum(int(torch.tensor(list(s)).prod().item()) * 4
                                     for s in transfer_shapes.values())

        return {
            "model_name": "vgg",
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


    def get_dummy_inputs(self, device='cpu', channels_last=True):
        if channels_last:
            dummy_inputs = (torch.rand(100, 3, 64, 64).contiguous(memory_format=torch.channels_last).to(device))
            # print(f'return dummy inputs: {dummy_inputs.reshape(-1, )[:1].item():.2f}, {dummy_inputs.reshape(-1, )[-1:].item():.2f}')
        else: dummy_inputs = (torch.rand(100, 3, 64, 64).to(device))
        return dummy_inputs

    def get_dummy_features(self, device='cpu', channels_last=True):
        if not channels_last:
            dummy_features_vgg = [
                torch.rand((100, 64, 64, 64)).to(device),
                torch.rand((100, 64, 64, 64)).to(device),
                torch.rand((100, 128, 32, 32)).to(device),
                torch.rand((100, 128, 32, 32)).to(device),
                torch.rand((100, 256, 16, 16)).to(device),
                torch.rand((100, 256, 16, 16)).to(device),
                torch.rand((100, 256, 16, 16)).to(device),
                torch.rand((100, 512, 8, 8)).to(device),
                torch.rand((100, 512, 8, 8)).to(device),
                torch.rand((100, 512, 8, 8)).to(device),
                torch.rand((100, 512, 4, 4)).to(device),
                torch.rand((100, 512, 4, 4)).to(device),
                torch.rand((100, 512, 4, 4)).to(device),
            ]
        else:
            dummy_features_vgg = [
                torch.rand((100, 64, 64, 64)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 64, 64, 64)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 128, 32, 32)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 128, 32, 32)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 256, 16, 16)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 256, 16, 16)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 256, 16, 16)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 512, 8, 8)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 512, 8, 8)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 512, 8, 8)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 512, 4, 4)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 512, 4, 4)).contiguous(memory_format=torch.channels_last).to(device),
                torch.rand((100, 512, 4, 4)).contiguous(memory_format=torch.channels_last).to(device),
            ]
        if hasattr(self, "connection_layers"):
            for i, layer in enumerate(self.connection_layers):
                if isinstance(layer, EmptyLayer):
                    dummy_features_vgg[i] = None
        elif hasattr(self, "output_feature_flag"):
            for i, flag in enumerate(self.output_feature_flag):
                if not flag:
                    dummy_features_vgg[i] = None

        return dummy_features_vgg

    def backbone_forward_eager_mode(self, shm_tensors, iter, verbose=False, inputs=None):
        time_line = {}
        check_value = '0' if iter == 11 else '0'
        chlast = getattr(self, "_channels_last", True)
        x = inputs if inputs is not None else self.get_dummy_inputs(device='cuda:0', channels_last=chlast)
        # print(f'iter {iter}: channel_last: {x.is_contiguous(memory_format=torch.channels_last)}, regular_channel: {x.is_contiguous()}')
        # if verbose: print(f'x: {tensor_hash(x)[-5:]}')
        send_names = [f'x']
        send_value(f'x', x, 'tensor', method=self.transfer_type, shared_memory=shm_tensors['x'], channels_last=chlast, check_value=check_value)
        time_line['backbone_start'] = time.time_ns()
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
                    if verbose: print(f'{key}: {tensor_hash(x)[-5:]}')
                    send_value(key, x, 'tensor', method=self.transfer_type, shared_memory=shm_tensors[key], channels_last=chlast, check_value=check_value)
                    time_line[f'backbone_{key}_trans1'] = time.time_ns()
                send_names.append(key)

            time_line[f'backbone_{key}_pooling0'] = time.time_ns()
            x = self.relu(x)
            if i in [1, 3, 6, 9, 12]: x = self.MaxPool2d(x)
            time_line[f'backbone_{key}_pooling1'] = time.time_ns()

            time_line[f'backbone_{key}_end'] = time.time_ns()


        time_line[f'backbone_end'] = time.time_ns()
        return time_line

    def side_forward_eager_mode(self, iter, verbose=False):
        # self.mismatch_count = {
        #     'spatial_match': [],
        #     'channel_match': [],
        # }
        time_line = {}
        chlast = getattr(self, "_channels_last", True)
        shapes = self.transfer_shapes
        check_value = '0' if iter in [11] else '0'
        # 1) recv input x
        time_line['side_x_recv0'] = time.time_ns()
        x_side_in = recv_value(f'x', 'tensor', shape=tuple(shapes['x']), method=self.transfer_type, channels_last=chlast, check_value=check_value)
        time_line['side_x_recv1'] = time.time_ns()
        time_line['side_start'] = time.time_ns()
        time_line['side_input_ladder_start'] = time_line['side_start']
        x_side = self.connection_layers[-1](x_side_in)
        time_line[f'side_input_ladder_end'] = time.time_ns()
        recv_names = [f'x']

        # 2) for each connection that is active, recv its backbone feature
        for i, layer in enumerate(self.side_layers):
            key = f"f{i}"
            layer_name = f'f{i}'
            time_line[f'side_{key}_start'] = time.time_ns()
            conn = self.connection_layers[i]
            if not isinstance(conn, EmptyLayer) and (layer_name in shapes): # not isinstance(conn, EmptyLayer) and
                # connection layer computation
                time_line[f'side_{key}_recv0'] = time.time_ns()
                feat_i = recv_value(key, 'tensor', shape=tuple(shapes[layer_name]),
                                    method=self.transfer_type, channels_last=chlast, check_value=check_value)
                time_line[f'side_{key}_recv1'] = time.time_ns()
                time_line[f'side_{key}_con0'] = time.time_ns()
                reduced_feature = conn(feat_i)
                if verbose: print(f'{key}: {tensor_hash(feat_i)[-5:]}')
                time_line[f'side_{key}_con1'] = time.time_ns()
                recv_names.append(key)

                # Addition: connection feature + side feature
                time_line[f'side_{key}_add0'] = time.time_ns()
                # reduced_feature = self._apply_connector(reduced_feature, x_side, getattr(self, 'layer_con_type', 'pool'), is_vit=False)
                # ui = torch.sigmoid(self.gate_params[i] / self.temperature)
                # x_side = ui * x_side + (1 - ui) * reduced_feature
                ui = torch.sigmoid(self.gate_params[i] / self.temperature)
                x_side.lerp_(reduced_feature, 1.0 - ui)
                time_line[f'side_{key}_add1']  = time.time_ns()

            # t2 = time.time_ns()
            time_line[f'side_{key}_layer0'] = time.time_ns()
            # x_side = self._coerce_x_to_layer_input(x_side, layer)
            x_side = layer(x_side)
            time_line[f'side_{key}_layer1'] = time.time_ns()

            if i in [1, 3, 6, 9, 12]:
                time_line[f'side_{key}_pooling0'] = time.time_ns()
                x_side = self.MaxPool2d(x_side)
                time_line[f'side_{key}_pooling1'] = time.time_ns()
            time_line[f'side_{key}_end'] = time.time_ns()

        for name in recv_names:
            ack(name)
            # close_shm(name, unlink=True)

        # 3) head
        time_line['side_head0'] = time.time_ns()
        x_side = self.avgpool_side(x_side)
        x_side = torch.flatten(x_side, 1)
        # x_side = self.dropout(x_side)
        x_side = self.upsample(x_side)
        # x_side = self.relu(x_side)
        # x_side = self.dropout(x_side)
        x_side = self.head(x_side)
        time_line['side_head1'] = time.time_ns()
        time_line[f'side_end'] = time.time_ns()
        # print(f'mismatch count: {self.mismatch_count}, channel_match sum: {sum(self.mismatch_count["channel_match"])}')
        return x_side, time_line

    def backbone_forward_eager_mode_no_transfer_latency(self, shm_tensors, iter, verbose=False, inputs=None):
        # Mirror of backbone_forward_eager_mode but with send_value_no_transfer for tensors,
        # so the publish handshake fires but no D->H copy occurs. Used for A_hat measurement.
        time_line = {}
        check_value = '0' if iter == 11 else '0'
        chlast = getattr(self, "_channels_last", True)
        x = inputs if inputs is not None else self.get_dummy_inputs(device='cuda:0', channels_last=chlast)
        send_names = [f'x']
        send_value_no_transfer(f'x', x, 'tensor', method=self.transfer_type, shared_memory=shm_tensors['x'], channels_last=chlast, check_value=check_value)
        time_line['backbone_start'] = time.time_ns()
        for i, layer in enumerate(self.backbone_layers):
            key = f"f{i}"
            time_line[f'backbone_{key}_start'] = time.time_ns()
            time_line[f'backbone_{key}_layer0'] = time.time_ns()
            x = layer(x)
            torch.cuda.synchronize()
            time_line[f'backbone_{key}_layer1'] = time.time_ns()
            if hasattr(self, "output_feature_flag") and self.output_feature_flag[i]:
                if self.transfer_type == 'shm':
                    time_line[f'backbone_{key}_trans0'] = time.time_ns()
                    if verbose: print(f'{key}: {tensor_hash(x)[-5:]}')
                    send_value_no_transfer(key, x, 'tensor', method=self.transfer_type, shared_memory=shm_tensors[key], channels_last=chlast, check_value=check_value)
                    time_line[f'backbone_{key}_trans1'] = time.time_ns()
                send_names.append(key)

            time_line[f'backbone_{key}_pooling0'] = time.time_ns()
            x = self.relu(x)
            if i in [1, 3, 6, 9, 12]: x = self.MaxPool2d(x)
            time_line[f'backbone_{key}_pooling1'] = time.time_ns()
            time_line[f'backbone_{key}_end'] = time.time_ns()

        time_line[f'backbone_end'] = time.time_ns()
        return time_line

    def side_forward_eager_mode_no_transfer_latency(self, iter, verbose=False):
        # Mirror of side_forward_eager_mode but with recv_value_no_transfer for tensors. The
        # tensors received are zeros of the expected shape; outputs are not numerically meaningful.
        time_line = {}
        chlast = getattr(self, "_channels_last", True)
        shapes = self.transfer_shapes
        check_value = '0' if iter in [11] else '0'
        time_line['side_x_recv0'] = time.time_ns()
        x_side_in = recv_value_no_transfer(f'x', 'tensor', shape=tuple(shapes['x']), method=self.transfer_type, channels_last=chlast, check_value=check_value)
        time_line['side_x_recv1'] = time.time_ns()
        time_line['side_start'] = time.time_ns()
        time_line['side_input_ladder_start'] = time_line['side_start']
        x_side = self.connection_layers[-1](x_side_in)
        time_line[f'side_input_ladder_end'] = time.time_ns()
        recv_names = [f'x']

        for i, layer in enumerate(self.side_layers):
            key = f"f{i}"
            layer_name = f'f{i}'
            time_line[f'side_{key}_start'] = time.time_ns()
            conn = self.connection_layers[i]
            if not isinstance(conn, EmptyLayer) and (layer_name in shapes):
                time_line[f'side_{key}_recv0'] = time.time_ns()
                feat_i = recv_value_no_transfer(key, 'tensor', shape=tuple(shapes[layer_name]),
                                                method=self.transfer_type, channels_last=chlast, check_value=check_value)
                time_line[f'side_{key}_recv1'] = time.time_ns()
                time_line[f'side_{key}_con0'] = time.time_ns()
                reduced_feature = conn(feat_i)
                if verbose: print(f'{key}: {tensor_hash(feat_i)[-5:]}')
                time_line[f'side_{key}_con1'] = time.time_ns()
                recv_names.append(key)

                time_line[f'side_{key}_add0'] = time.time_ns()
                ui = torch.sigmoid(self.gate_params[i] / self.temperature)
                x_side.lerp_(reduced_feature, 1.0 - ui)
                time_line[f'side_{key}_add1']  = time.time_ns()

            time_line[f'side_{key}_layer0'] = time.time_ns()
            x_side = layer(x_side)
            time_line[f'side_{key}_layer1'] = time.time_ns()

            if i in [1, 3, 6, 9, 12]:
                time_line[f'side_{key}_pooling0'] = time.time_ns()
                x_side = self.MaxPool2d(x_side)
                time_line[f'side_{key}_pooling1'] = time.time_ns()
            time_line[f'side_{key}_end'] = time.time_ns()

        for name in recv_names:
            ack(name)

        time_line['side_head0'] = time.time_ns()
        x_side = self.avgpool_side(x_side)
        x_side = torch.flatten(x_side, 1)
        x_side = self.upsample(x_side)
        x_side = self.head(x_side)
        time_line['side_head1'] = time.time_ns()
        time_line[f'side_end'] = time.time_ns()
        return x_side, time_line

    def convert_to_backbone_model(self, device, real_transfer=True):
        # mark which backbone features we need to emit
        self.output_feature_flag = []
        for connection_layer in self.connection_layers:
            self.output_feature_flag.append(not isinstance(connection_layer, EmptyLayer))

        # get transfer tensor's name and shape
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

        # drop side parts for backbone-only
        del self.side_layers, self.connection_layers, self.gate_params, self.upsample, self.head
        self._channels_last = True  # backbone stream uses channels_last
        self._real_transfer = bool(real_transfer)
        self.forward = self.backbone_forward_eager_mode if self._real_transfer else self.backbone_forward_eager_mode_no_transfer_latency
        for layer in self.backbone_layers: layer.to(device)

    def convert_to_side_model(self, device, real_transfer=True):
        # keep side modules on CPU (TEE)
        del self.backbone_layers
        self.side_device_init(device)

        # predefine transfer shapes so recv_value(type='tensor') can reshape
        shapes = {'x': tuple(self.get_dummy_inputs(device='cpu', channels_last=True).shape)}

        feats = self.get_dummy_features(device='cpu', channels_last=True) or []
        num_feat = len(feats)  # backbone-aligned features only (no input-ladder)

        for i, conn in enumerate(self.connection_layers):
            # Skip input-ladder (and any extra trailing connections beyond feats)
            if i >= num_feat:continue
            # Only add a slot if this connection is active and the feature exists
            if (not isinstance(conn, EmptyLayer)) and feats[i] is not None:
                shapes[f"f{i}"] = tuple(feats[i].shape)

        self.transfer_shapes = shapes
        self._channels_last = True
        self._real_transfer = bool(real_transfer)
        self.forward = self.side_forward_eager_mode if self._real_transfer else self.side_forward_eager_mode_no_transfer_latency




# Functional Test
if __name__ == '__main__':

    # Define the model
    model = VGGNet()
    print(model)

    # Get the model size
    print(model.get_model_size())
