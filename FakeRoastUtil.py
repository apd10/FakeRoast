import torch
import torch.nn as nn
from .FakeRoast import *
import copy 
import numpy as np

ROAST_INIT=0.04

def roast(pytorch_model, is_global, compression=None, memory_mb=None, max_params=None):
    assert((not is_global)  or (memory_mb is not None or max_params is not None))
    assert((is_global) or (compression))

    init_std = None
    roast_array = None

    if is_global:
        if memory_mb is not None:
            max_params = int(memory_mb * 1000000 / 4)

        init_std = ROAST_INIT
        roast_array = nn.Parameter(torch.FloatTensor(max_params).uniform_(-init_std, init_std))
        
    seed = 1
    rzmodel = copy.deepcopy(pytorch_model)
    _roast("head", rzmodel, is_global, roast_array, init_std, compression, seed, chunk_size=32)
    return rzmodel

def _roast(name, pytorch_model, is_global, roast_array, init_std, compression, init_seed, chunk_size):
    seed = init_seed * 1024
    #print(name, "->", type(pytorch_model))


    for attr in dir(pytorch_model):
        target_attr = getattr(pytorch_model, attr)
        #print(name, "->", attr, "type:", type(target_attr), target_attr)
        if type(target_attr) == torch.nn.modules.Linear:
            seed = seed + 1
            new_attr = FakeRoastLinear(target_attr.in_features, 
                                       target_attr.out_features,
                                       target_attr.bias is not None,
                                       is_global,
                                       roast_array,
                                       ROAST_INIT,
                                       compression,
                                       False,
                                       "random",
                                       seed)
            print("replaced", target_attr)
            setattr(pytorch_model, attr, new_attr)
        elif type(target_attr) == torch.nn.modules.sparse.Embedding:
            seed = seed + 1
            new_attr = FakeRoastEmbedding(target_attr.num_embeddings, 
                                          target_attr.embedding_dim,
                                          is_global, roast_array, ROAST_INIT, 
                                          compression, target_attr.padding_idx, 
                                          target_attr.max_norm, target_attr.norm_type,
                                          target_attr.scale_grad_by_freq, 
                                          target_attr.sparse)
            print("replaced", target_attr)
            setattr(pytorch_model, attr, new_attr)
        elif type(target_attr) == torch.nn.modules.Conv2d:
            seed = seed + 1
            new_attr = FakeRoastConv2d( target_attr.in_channels,
                              target_attr.out_channels,
                              target_attr.kernel_size,
                              is_global,
                              roast_array,
                              ROAST_INIT,
                              compression,
                              target_attr.stride,
                              target_attr.padding,
                              target_attr.dilation,
                              target_attr.groups, 
                              target_attr.bias is not None,
                              target_attr.padding_mode,
                              False,
                              "random",
                              seed)
            print("replaced", target_attr)
            setattr(pytorch_model, attr, new_attr)
        
    for name, immediate_child_module in  pytorch_model.named_children():
        target_attr = immediate_child_module
        if type(immediate_child_module) in [torch.nn.modules.Linear , torch.nn.modules.linear.Linear]:
            seed = seed + 1
            new_attr = FakeRoastLinear(target_attr.in_features, 
                                       target_attr.out_features,
                                       target_attr.bias is not None,
                                       is_global,
                                       roast_array,
                                       ROAST_INIT,
                                       compression,
                                       False,
                                       "random",
                                       seed)
            print("replaced", target_attr)
            setattr(pytorch_model, name, new_attr)
        elif type(immediate_child_module) == torch.nn.modules.sparse.Embedding:
            seed = seed + 1
            new_attr = FakeRoastEmbedding(target_attr.num_embeddings, 
                                          target_attr.embedding_dim,
                                          is_global, roast_array, ROAST_INIT, 
                                          compression, target_attr.padding_idx, 
                                          target_attr.max_norm, target_attr.norm_type,
                                          target_attr.scale_grad_by_freq, 
                                          target_attr.sparse)
            print("replaced", target_attr)
            setattr(pytorch_model, name, new_attr)
        elif type(immediate_child_module) == torch.nn.modules.Conv2d:
            seed = seed + 1
            new_attr = FakeRoastConv2d( target_attr.in_channels,
                              target_attr.out_channels,
                              target_attr.kernel_size,
                              is_global,
                              roast_array,
                              ROAST_INIT,
                              compression,
                              target_attr.stride,
                              target_attr.padding,
                              target_attr.dilation,
                              target_attr.groups, 
                              target_attr.bias is not None,
                              target_attr.padding_mode,
                              False,
                              "random",
                              seed)
            print("replaced", target_attr)
            setattr(pytorch_model, name, new_attr)
        init_seed = init_seed + 1
        _roast(name, immediate_child_module, is_global, roast_array, init_std, compression, init_seed, chunk_size)
