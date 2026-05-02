# Description: This file contains the implementation of the MixEdge.

from modules.layers import *

def build_candidate_ops_cfg(candidate_cfg, reduction_factor_candidates, archi_type):
    ops = []
    for rf in reduction_factor_candidates: # iterate rf first then candidates_ops, otherwise, wrong config when applying genetic algorithm.
        for name in candidate_cfg:
            if isinstance(name, str) and name == 'empty': # apply mapping for simple config
                layer = EmptyLayer(0, 0, rf, type=archi_type)
            elif isinstance(name, dict):
                if name['type'].lower() == 'conv2d':
                    layer = ReducedConvLayer(name['in_channels'], name['out_channels'], kernel_size=name['kernel_size'], padding=name['padding'], stride=name['stride'],
                                             reduction_factor=rf, type=archi_type, bias=name['bias'], act_func=name['act_func'] if 'act_func' in name else None, use_bn=name.get('use_bn', False)) # , use_bn=name.get('use_bn', False)
                elif name['type'].lower() == 'linear':
                    layer = ReducedLinearLayer(name['in_features'], name['out_features'], bias=name['bias'], type=archi_type, reduction_factor=rf)
                elif name['type'].lower() == 't5-enc-block':
                    d = name['embed_dim'] // rf
                    if d % name['num_heads'] != 0:
                        raise ValueError(f'The dimension {d} of embedding must be divisible by number of heads {name["num_heads"]}')
                    layer = ReducedT5Block(name['embed_dim'], name['num_heads'], reduction_factor=rf, type='encoder')
                elif name['type'].lower() == 't5-dec-block':
                    d = name['embed_dim'] // rf
                    if d % name['num_heads'] != 0:
                        raise ValueError(f'The dimension {d} of embedding must be divisible by number of heads {name["num_heads"]}')
                    layer = ReducedT5Block(name['embed_dim'], name['num_heads'], reduction_factor=rf, type='decoder')
                elif name['type'].lower() == 'vit-block':
                    layer = ViTBlock(name['embed_dim'], name['num_heads'], reduction_factor=rf)
                elif name['type'].lower() == 'llama-block':
                    d = name['embed_dim'] // rf
                    if d < 1:
                        raise ValueError(f"llama-block: dim_side={d} too small for rf={rf}")
                    layer = LlamaBlock(name['embed_dim'], name['num_heads'], reduction_factor=rf)
                elif 'lora' in name['type'].lower() or 'adapter' in name['type'].lower():
                    layer = build_layer_from_config(rf, name)
                else:
                    raise ValueError(f'{name} is not a string or dict')

            ops.append(layer)

    return ops

class MixedEdge_NoParams(nn.Module):
    def __init__(self, candidate_ops, reduction=None):
        super(MixedEdge_NoParams, self).__init__()
        self.candidate_ops = nn.ModuleList(candidate_ops)
        self.n_choices = len(candidate_ops)
        self.active_index = [0]  # GA selects this explicitly

        # Optional: remember the previous individual or evaluation
        self.fitness = None
        self.reduction = reduction


    @property
    def chosen_op(self):
        return self.candidate_ops[self.active_index[0]]

    @property
    def random_op(self):
        index = np.random.choice(range(self.n_choices))
        return self.candidate_ops[index]

    @property
    def active_op(self):
        return self.candidate_ops[self.active_index[0]]


    def set_chosen_op_active(self):
        chosen_index, _ = self.chosen_index
        self.active_index = [chosen_index]
        self.inactive_index = [_i for _i in range(0, chosen_index)] + \
                                [_i for _i in range(chosen_index + 1, self.n_choices)]

    def forward(self, x):
        if self.active_op is None: return None
        return self.active_op(x)

    def is_zero_layer(self):
        if self.active_op is None: return False
        return self.active_op.is_zero_layer()

    def get_model_size(self):
        if self.active_op is None: return 0
        return self.active_op.get_model_size()

    def binarize(self, sample):
        # binarize based on config
        self.active_index = [sample]
        self.inactive_index = [_i for _i in range(0, sample)] + \
                                [_i for _i in range(sample + 1, self.n_choices)]

    @property
    def config(self):
        return {
            'name': self.__class__.__name__,
            'n_choices': self.n_choices,
            'active_index': self.active_index,
        }

    # -------- Genetic Operators -------- #

    def mutate(self, mutation_rate=0.1):
        if random.random() < mutation_rate:
            old_idx = self.active_index
            candidates = list(range(self.n_choices))
            candidates.remove(old_idx)
            self.active_index = random.choice(candidates)

    def set_active_index(self, index):
        assert 0 <= index < self.n_choices
        self.active_index = index

