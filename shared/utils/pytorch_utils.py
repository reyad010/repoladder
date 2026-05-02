# Description: This file contains some utility functions.
import torch.nn as nn
import importlib
import random, torch
import numpy as np
from collections import OrderedDict


def build_activation(act_func, inplace=True):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    logsoftmax = nn.LogSoftmax()
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))


class ShuffleLayer(nn.Module):
    def __init__(self, groups):
        super(ShuffleLayer, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        # noinspection PyUnresolvedReferences
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / float(self.count)


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # View VS Reshape !!!
        # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def delta_ij(i, j):
    if i == j:
        return 1
    else:
        return 0


def get_split_list(in_dim, child_num):
    in_dim_list = [in_dim // child_num] * child_num
    for _i in range(in_dim % child_num):
        in_dim_list[_i] += 1
    return in_dim_list

def L2NormRegularizer(layer1, layer2, strength):
    layer1_weight = layer1.weight
    layer2_weight = layer2.weight

    # Determine the maximum size for each dimension
    max_size = [max(layer1_weight.size(i), layer2_weight.size(i)) for i in range(4)]

    # Create new tensors by padding to match the maximum size in each dimension
    padded_layer1_weight = torch.nn.functional.pad(layer1_weight, 
                                                   (0, max_size[3] - layer1_weight.size(3),  # pad width
                                                    0, max_size[2] - layer1_weight.size(2),  # pad height
                                                    0, max_size[1] - layer1_weight.size(1),  # pad channels
                                                    0, max_size[0] - layer1_weight.size(0))) # pad batch
    
    padded_layer2_weight = torch.nn.functional.pad(layer2_weight, 
                                                   (0, max_size[3] - layer2_weight.size(3),  # pad width
                                                    0, max_size[2] - layer2_weight.size(2),  # pad height
                                                    0, max_size[1] - layer2_weight.size(1),  # pad channels
                                                    0, max_size[0] - layer2_weight.size(0))) # pad batch
    
    # Compute the L2 norm difference between the padded weights
    weight_diff = torch.norm(padded_layer1_weight - padded_layer2_weight, p=2)
    
    # Calculate the range difference between the padded weights
    range_diff = torch.abs(torch.max(padded_layer1_weight) - torch.max(padded_layer2_weight)) + \
                 torch.abs(torch.min(padded_layer1_weight) - torch.min(padded_layer2_weight))
    
    # Compute the regularization loss
    regularization_loss = (weight_diff + range_diff) * strength

    return regularization_loss

def summarize_trainable_ratio(model: nn.Module, verbose=True):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = 100.0 * trainable_params / total_params if total_params > 0 else float('nan')
    trainable_layer_names = []
    if verbose:
        for name, params in model.named_parameters():
            if params.requires_grad:
                trainable_layer_names.append(name)
                # print(f'trainable layer: {name}, params: {params.numel()}')

        print(f"{'Trainable layer (3):':<20} {trainable_layer_names}")
        print(f"{'Total params:':<20} {total_params / 1e6:.2f} M")
        print(f"{'Trainable params:':<20} {trainable_params / 1e6:.2f} M")
        print(f"{'Trainable %:':<20} {ratio:.2f}%")
    return total_params, trainable_params, ratio

def load_lst_net(model_name):
    try:
        module = importlib.import_module(f"models.lst_{model_name}.ladder_side_tune_model")
        return module.LadderSideTuneNet
    except ModuleNotFoundError as e:
        raise ImportError(f"No obfuscated model found for '{model_name}'") from e

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pass

def fisher_to_lst_model(model):
    import torchvision
    from st_security.pruning.fisher import compute_fisher_cv
    from st_security.pruning.pruning_methods import pruning_v2_cv
    importance_measure = None #compute_fisher_cv(torchvision.models.vgg16(pretrained=True).to(device), self.run_manager.run_config.calc_loader,1024)
    # vgg16side.state_dict()
    def rename_side_state_dict(model, state_dict_keys):
        new_state_dict = OrderedDict()
        if model.model_name == 'vgg':
            side_key = []
            for i in range(len(model.side_layers)):
                side_key.append(f'side_layers.{i}.conv.weight')
                side_key.append(f'side_layers.{i}.conv.bias')

            for key1, key2 in zip(side_key, state_dict_keys):
                if key1 in model.state_dict().keys():
                    new_state_dict[key2] = model.state_dict()[key1]
            return new_state_dict

        else: raise NotImplementedError

    pytorch_st_keys = torchvision.models.vgg16(pretrained=True).state_dict().keys()
    new_state_dict = rename_side_state_dict(model, pytorch_st_keys)
    pruned_state_dict = pruning_v2_cv(model.backbone_layers, new_state_dict,importance_measure=importance_measure)
    out = model.side_layers.load_state_dict_local(pruned_state_dict, strict=False)
    print(out)

    return model


def extra_cfg_for_model(model_name, type='official'):
    import torchvision
    model = torchvision.models.vgg16(pretrained=True)
    if type == 'official':
        def get_moedel_size(val1):
            return 138000000
        def weight_parameters(self):
            """
            Yield only parameters whose names contain 'weight'.
            Remove the if-clause if you really want *all* params.
            """
            for name, param in self.named_parameters():
                    yield param
        import types
        model.weight_parameters = types.MethodType(weight_parameters, model)
        model.get_model_size = types.MethodType(get_moedel_size, model)
        model.model_name = model_name
        model.freeze_params = types.MethodType(get_moedel_size, model)
    elif type == 'lora':
        model.classifier = torch.nn.Linear(512, 10)
        model.classifier.requires_grad = True
        from st_security.utils_2 import inject_lora_conv
        inject_lora_conv(model.features, r=16, dropped_layers=[])
        def get_moedel_size(val1):
            return 138000000
        def weight_parameters(self):
            """
            Yield only parameters whose names contain 'weight'.
            Remove the if-clause if you really want *all* params.
            """
            for name, param in self.named_parameters():
                yield param

        import types
        model.weight_parameters = types.MethodType(weight_parameters, model)
        model.get_model_size = types.MethodType(get_moedel_size, model)
        model.model_name = model_name
        model.freeze_params = types.MethodType(get_moedel_size, model)
    else: raise NotImplementedError




    return model