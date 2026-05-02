
from queue import Queue
import copy

from modules.mix import *
from models.lst_vit.base_model import *

class LadderSideTuneNet(ViT):

    def __init__(self, search_space, num_classes):
        self._redundant_modules = None
        self._unused_modules = None
        self.search_space = search_space
        search_space_connection = search_space['connection']
        search_space_side= search_space['side']
        reduction_factor_candidates = search_space['reduction_factor']
        top_config = {
              "type": "VisionTransformer",
              "img_size": [
                224,
                224
              ],
              "patch_size": [
                16,
                16
              ],
              "in_chans": 3,
              "embed_dim": 768,
              "depth": 12,
              "num_heads": 12
            }

        # generate side/connection config

        connection_config = [{
            "type": 'Linear',
            "in_features": top_config['embed_dim'],
            "out_features": top_config['embed_dim'],
            "bias": False,
        } for _ in range(top_config['depth']+1)] # 13 connection layers

        side_config = [
            {
                "type": 'vit-block',
                'num_heads': top_config['num_heads'],
                'embed_dim': top_config['embed_dim'],
            } for _ in range(top_config['depth'])
        ]



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

        side_cfg= []
        for s_cfg in side_config:
            tmp = []
            for name in search_space_side:
                if name == 'same':
                    tmp.append(s_cfg) # only Conv3x3 in vgg models
                else: tmp.append(name)
            side_cfg.append(copy.deepcopy(tmp))

        upsample_cfg = [
            {
                "type": "Linear",
                "in_features": 768,
                "out_features": 768,
                "bias": True
            }
        ]

        layers_dict = {
            'connection_layers': [],
            'side_layers': [],
            'head': None,
            'upsample': None,
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

        layers_dict['head'] = nn.Linear(768, num_classes) # the head is fixed.


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
                elif isinstance(cur_layer, ViTBlock):
                    side_layers.append(cur_layer)
                else:
                    raise NotImplementedError


        upsample = self.upsample.conv.candidate_ops[up_cfg[0]]
        layers_dict =  {
                'connection_layers': connection_layers,
                'side_layers': side_layers,
                'upsample': upsample,
                'head': self.head,
            }
        normal_net = ViT(layers_dict,self.num_classes)
        for gp_src, gp_dst in zip(self.gate_params, normal_net.gate_params):
            gp_dst.data.copy_(gp_src.data)

        return normal_net