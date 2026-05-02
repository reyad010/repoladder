from modules.mix import *
from models.lst_resnet.base_model import *

class LadderSideTuneNet(ResNet):

    def __init__(self, search_space, num_classes):
        self._redundant_modules = None
        self._unused_modules = None
        self.search_space = search_space
        search_space_connection = search_space['connection']
        search_space_side = search_space['side']
        reduction_factor_candidates = search_space['reduction_factor']

        conv_cfg = [{"type": "Conv2d", 'in_channels': 3, 'out_channels': 64, 'kernel_size': (7, 7), 'padding': (3, 3), 'stride': (2, 2), 'bias': False, 'use_bn': True, 'act_func': 'relu'},
                    {"type": "Conv2d", 'in_channels': 64, 'out_channels': 64, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': False, 'use_bn': True, 'act_func': 'relu'},
                    {"type": "Conv2d", 'in_channels': 64, 'out_channels': 64, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': False, 'use_bn': True, 'act_func': None},
                    {"type": "Conv2d", 'in_channels': 64, 'out_channels': 64, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': False, 'use_bn': True, 'act_func': 'relu'},
                    {"type": "Conv2d", 'in_channels': 64, 'out_channels': 64, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': False, 'use_bn': True, 'act_func': None},
                    {"type": "Conv2d", 'in_channels': 64, 'out_channels': 128, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (2, 2), 'bias': False, 'use_bn': True, 'act_func': 'relu'},
                    {"type": "Conv2d", 'in_channels': 128, 'out_channels': 128, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': False, 'use_bn': True, 'act_func': None},
                    {"type": "Conv2d", 'in_channels': 128, 'out_channels': 128, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': False, 'use_bn': True, 'act_func': 'relu'},
                    {"type": "Conv2d", 'in_channels': 128, 'out_channels': 128, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': False, 'use_bn': True, 'act_func': None},
                    {"type": "Conv2d", 'in_channels': 128, 'out_channels': 256, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (2, 2), 'bias': False, 'use_bn': True, 'act_func': 'relu'},
                    {"type": "Conv2d", 'in_channels': 256, 'out_channels': 256, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': False, 'use_bn': True, 'act_func': None},
                    {"type": "Conv2d", 'in_channels': 256, 'out_channels': 256, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': False, 'use_bn': True, 'act_func': 'relu'},
                    {"type": "Conv2d", 'in_channels': 256, 'out_channels': 256, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': False, 'use_bn': True, 'act_func': None},
                    {"type": "Conv2d", 'in_channels': 256, 'out_channels': 512, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (2, 2), 'bias': False, 'use_bn': True, 'act_func': 'relu'},
                    {"type": "Conv2d", 'in_channels': 512, 'out_channels': 512, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': False, 'use_bn': True, 'act_func': None},
                    {"type": "Conv2d", 'in_channels': 512, 'out_channels': 512, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': False, 'use_bn': True, 'act_func': 'relu'},
                    {"type": "Conv2d", 'in_channels': 512, 'out_channels': 512, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': False, 'use_bn': True, 'act_func': None}]

        downsample_cfg = [{"type": "Conv2d", 'in_channels': 64, 'out_channels': 128, 'kernel_size': (1, 1), 'padding': (0, 0), 'stride': (2, 2), 'bias': False, 'use_bn': True, 'act_func': None},
                          {"type": "Conv2d", 'in_channels': 128, 'out_channels': 256, 'kernel_size': (1, 1), 'padding': (0, 0), 'stride': (2, 2), 'bias': False, 'use_bn': True, 'act_func': None},
                          {"type": "Conv2d", 'in_channels': 256, 'out_channels': 512, 'kernel_size': (1, 1), 'padding': (0, 0), 'stride': (2, 2), 'bias': False, 'use_bn': True, 'act_func': None}]

        B_out_channels = [cfg['out_channels'] for cfg in conv_cfg]
        B_in_channels = [cfg['in_channels'] for cfg in conv_cfg]

        S_in_channels = B_in_channels[1:] + B_out_channels[-1:]
        S_out_channels = B_out_channels[1:] + B_out_channels[-1:]

        connection_config = [{
            "type": 'Conv2d',
            "in_channels": cfg['out_channels'],
            "out_channels": cfg['out_channels'],  # both set to out_channels
            "kernel_size": [1, 1],
            "stride": [1, 1],
            "padding": [0, 0],
            "bias": False, #True,
            "act_func": None, # 'relu', #
            "use_bn": False,
        } for cfg in conv_cfg]

        connection_config.append(conv_cfg[0].copy())  # for the input connection layer
        connection_config[-1]['bias'] = connection_config[0]['bias']  # 'relu' #
        connection_config[-1]['act_func'] = connection_config[0]['act_func'] # 'relu' #
        connection_config[-1]['use_bn'] = connection_config[0]['use_bn'] # False



        # side_config_conv = [{**cfg, 'in_channels': in_c, 'out_channels': out_c,} for cfg, in_c, out_c in zip(conv_cfg, S_in_channels, S_out_channels)]

        side_config_conv = conv_cfg[1:]
        side_config_conv.append({"type": "Conv2d", 'in_channels': 512, 'out_channels': 512,
                                 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1),
                                 'bias': False, 'use_bn': True, 'act_func': None})


        side_config = side_config_conv + downsample_cfg

        # mapping
        con_cfg = []
        for c_cfg in connection_config:
            tmp = []
            for name in search_space_connection:
                if name == 'regular':
                    tmp.append(c_cfg)
                elif name == 'lora' or name == 'adapter':
                    new_cfg = copy.deepcopy(c_cfg)
                    new_cfg.update({
                        "type": c_cfg['type'] + '.' + name,
                        "rank": 16,
                    })
                    tmp.append(new_cfg)
                else:
                    tmp.append(name)
            con_cfg.append(copy.deepcopy(tmp))

        side_cfg = []
        for s_cfg in side_config:  # 13 conv
            tmp = []
            for name in search_space_side:
                if name == 'same':
                    tmp.append(s_cfg)  # only Conv3x3 in vgg models
                else:
                    tmp.append(name)
            side_cfg.append(copy.deepcopy(tmp))

        upsample_cfg = [
            {
                "type": "Linear",
                "in_features": 512,
                "out_features": 512,
                "bias": True
            }
        ]
        layers_dict = {
            'connection_layers': [],
            'side_layers': [],
            'upsample': None,
            'head': None,
        }

        # build connections layers:
        for candidate_con_cfg in con_cfg:

            candidates_op = build_candidate_ops_cfg(candidate_con_cfg, reduction_factor_candidates, archi_type='connection')

            connection_layer = MixedEdge_NoParams(candidate_ops=candidates_op)

            block = ObfuscationBlock(connection_layer)
            layers_dict['connection_layers'].append(block)

        # build side nets
        for candidate_side_cfg in side_cfg:
            candidates_op = build_candidate_ops_cfg(candidate_side_cfg, reduction_factor_candidates, archi_type='side')
            side_layer = MixedEdge_NoParams(candidate_ops=candidates_op)
            block = ObfuscationBlock(side_layer)
            layers_dict['side_layers'].append(block)

        # build upsample
        candidates_op = build_candidate_ops_cfg(upsample_cfg, reduction_factor_candidates, archi_type='upsample')
        upsample = MixedEdge_NoParams(candidate_ops=candidates_op)
        block = ObfuscationBlock(upsample)
        layers_dict['upsample'] = block

        layers_dict['head'] = nn.Linear(512, num_classes)  # the head is fixed.


        super(LadderSideTuneNet, self).__init__(layers_dict, num_classes)

    def convert_to_normal_net(self, ladder_config):
        conn_len = len(self.connection_layers)
        side_len = len(self.side_layers)

        conn_cfg = ladder_config[:conn_len]
        side_cfg = ladder_config[conn_len:conn_len + side_len]
        up_cfg = ladder_config[conn_len + side_len:]
        connection_layers = []
        side_layers = []
        upsample = None

        # Replace MixedEdge in connection_layers
        for i, choice in enumerate(conn_cfg):
            if isinstance(self.connection_layers[i], ObfuscationBlock):
                cur_layer = self.connection_layers[i].conv.candidate_ops[choice]
                if isinstance(cur_layer, ReducedGroupingLayer) or isinstance(cur_layer, EmptyLayer):
                    connection_layers.append(cur_layer)
                elif isinstance(cur_layer, LALinear) or isinstance(cur_layer, LAConv2d):
                    connection_layers.append(cur_layer)
                elif isinstance(cur_layer, ReducedConvLayer) or isinstance(cur_layer, ReducedLinearLayer):
                    connection_layers.append(cur_layer)
                else:  # ReducedConv
                    raise NotImplementedError
                    seq = nn.Sequential()
                    if hasattr(cur_layer, 'conv'):
                        seq.add_module('conv', cur_layer.conv)
                    if hasattr(cur_layer, 'linear'):
                        seq.add_module('linear', cur_layer.linear)
                    if hasattr(cur_layer, 'bn'):
                        seq.add_module('bn', cur_layer.bn)
                    if hasattr(cur_layer, 'activation'):
                        seq.add_module('activation', cur_layer.activation)

                    seq = unwrap_sequential_if_single(seq)

                    connection_layers.append(seq)

        # Replace MixedEdge in side_layers
        for i, choice in enumerate(side_cfg):
            if isinstance(self.side_layers[i], ObfuscationBlock):
                cur_layer = self.side_layers[i].conv.candidate_ops[choice]
                if isinstance(cur_layer, ReducedGroupingLayer) or isinstance(cur_layer, EmptyLayer):
                    side_layers.append(cur_layer)
                elif isinstance(cur_layer, LALinear) or isinstance(cur_layer, LAConv2d):
                    side_layers.append(cur_layer)
                elif isinstance(cur_layer, ReducedConvLayer) or isinstance(cur_layer, ReducedLinearLayer):
                    side_layers.append(cur_layer)
                else:
                    raise NotImplementedError
                    seq = nn.Sequential()
                    if hasattr(cur_layer, 'conv'):
                        seq.add_module('conv', cur_layer.conv)
                    if hasattr(cur_layer, 'linear'):
                        seq.add_module('linear', cur_layer.linear)
                    if hasattr(cur_layer, 'bn'):
                        seq.add_module('bn', cur_layer.bn)
                    if hasattr(cur_layer, 'activation'):
                        seq.add_module('activation', cur_layer.activation)

                    seq = unwrap_sequential_if_single(seq)

                    side_layers.append(seq)

        upsample = self.upsample.conv.candidate_ops[up_cfg[0]]
        layers_dict = {
                'connection_layers': connection_layers,
                'side_layers': side_layers,
                'upsample': upsample,
                'head': self.head,
            }
        normal_net = ResNet(layers_dict, self.num_classes)
        for gp_src, gp_dst in zip(self.gate_params, normal_net.gate_params):
            gp_dst.data.copy_(gp_src.data)

        return normal_net