'''
  why new version? Previous version was more of a script with huge technical debt.
  goals
      - modularize the script .. separate the roast stuff from the model parser
      - run by sparsity
      - apply sparisty to only compressed modules
      - add a grad_scaler to roast-array (new logic to be implemeted next)
      - return a summary of what is compressed and how

'''
import torch
from .FakeRoast import *
#from FakeRoast import *

# general functions
NONE=0
INFO=1
DEBUG=2

def get_module_params(target_attr):
    ns = 0
    for p in target_attr.parameters():
        if p.requires_grad :
            ns += p.numel()
    return ns

class ModelParser:

    def __init__(self):
        self.verbose = NONE
        pass
  
    def lambda_init(self, state_dict):
        return state_dict

    def lambda_func(self, state_dict):
        return state_dict

    def lambda_next(self, state_dict):
        return state_dict

    def run(self, name, model, state_dict):
        state_dict['model'] = model
        state_dict = self.lambda_init(state_dict)
        for attr in dir(model):
            target_attr = getattr(model, attr)
            state_dict['target_attr'] = target_attr
            state_dict['name'] = attr
            state_dict['model'] = model
            state_dict = self.lambda_func(state_dict)
        for name, immediate_child_module in  model.named_children():
            target_attr = immediate_child_module
            state_dict['target_attr'] = target_attr
            state_dict['name'] = name
            state_dict['model'] = model
            state_dict = self.lambda_func(state_dict)
            state_dict = self.lambda_next(state_dict)
            self.run(name, immediate_child_module, state_dict)

class ModelPrinter(ModelParser):
    def __init__(self, model):
        super(ModelPrinter, self).__init__()
        self.model = model

    def lambda_func(self, state_dict):
        print("--->", type(state_dict['target_attr']), isinstance(state_dict['target_attr'], torch.nn.Module))
        return state_dict

    def process(self):
        self.run("model", self.model, {})

class Roastable:
    def __init__(self, module_limit_size=None, verbose=NONE):
        self.LINEAR = [torch.nn.Linear, torch.nn.modules.Linear]
        self.CONV2D = [torch.nn.Conv2d, torch.nn.modules.Conv2d]
        self.EMBEDDING = [torch.nn.Embedding, torch.nn.modules.Embedding]
        self.module_limit_size=module_limit_size
        self.verbose=verbose

    def is_linear(self, attr):
        return type(attr) in self.LINEAR

    def is_conv2d(self, attr):
        return type(attr) in self.CONV2D

    def is_embedding(self, attr):
        return type(attr) in self.EMBEDDING

    def roastable(self, attr):

        #if the module has been marked as do not roast
        do_not_roast = False
        if 'do_not_roast' in dir(attr):
            do_not_roast = attr.do_not_roast

        # checks
        sanity_checks = True
        if self.module_limit_size is not None and isinstance(attr, torch.nn.Module):
            sanity_checks = sanity_checks and (get_module_params(attr) >= self.module_limit_size)
            if self.verbose > DEBUG:
                print("checker", attr, get_module_params(attr), self.module_limit_size)

        # modules
        module_check = (self.is_linear(attr) 
                                  or self.is_conv2d(attr) 
                                  or self.is_embedding(attr))
        return (not do_not_roast) and sanity_checks and  module_check

    def get_parameter(self, attr):
        assert(self.roastable(attr))
        idc = id(attr.weight)
        c = attr.weight.numel()
        return idc, c
        
class ModelRoastableParameters(ModelParser, Roastable):
    def __init__(self, model, module_limit_size=None, verbose=NONE):
        ModelParser.__init__(self)
        Roastable.__init__(self, module_limit_size=module_limit_size, verbose=verbose)
        self.model = model

    def lambda_func(self, state_dict):
        attr = state_dict['target_attr']
        if isinstance(attr, torch.nn.Parameter):
            if attr.requires_grad:
                state_dict['all'][id(attr)] = attr.numel()

        if isinstance(attr, torch.nn.Module):
            is_roastable = self.roastable(attr)
            if is_roastable:
                  idc, c = self.get_parameter(attr)
                  state_dict['compressable'][idc] = c

        return state_dict

    def process(self):
        state_dict = {"compressable" : {}, "all" : {}}
        self.run("model", self.model, state_dict)
        
        total = 0
        roastable = 0
        for _,i in state_dict['compressable'].items():
            roastable += i
        for _,i in state_dict['all'].items():
            total += i
    
        #print("Roastable {} / {}".format(roastable, total))
        state_dict['roastable'] = roastable
        state_dict['all'] = total

        return state_dict



class ModelRoaster(ModelParser, Roastable):

    def __init__(self, model, roast_global, sparsity, module_limit_size=None, verbose=NONE):
        ModelParser.__init__(self)
        Roastable.__init__(self, module_limit_size=module_limit_size, verbose=verbose)
      
        self.verbose = verbose

        self.ROAST_INIT = 0.04
        self.model = model
        self.is_global = roast_global
        self.compression = sparsity

        parameter_finder = ModelRoastableParameters(model, module_limit_size=module_limit_size)
        pf = parameter_finder.process()
        roastable_params, total_params = pf['roastable'], pf['all']

        if self.verbose >= INFO:
            print("Roastable params: {}/{}".format(roastable_params, total_params))

        if self.is_global:
            ''' need to compute the max params: sparsity is applied to roastable parameters '''
            max_params = int(sparsity * roastable_params)
            self.roast_array = torch.nn.Parameter(torch.FloatTensor(max_params).uniform_(-self.ROAST_INIT, self.ROAST_INIT))
        else:
            self.roast_array = None

        self.original_total_params = total_params
        self.original_roastable_params = roastable_params

    def make_roast_module(self, target_attr, seed):
        if not self.roastable(target_attr):
              return None
        new_attr = None
        if self.is_linear(target_attr):
            new_attr = FakeRoastLinear(target_attr.in_features, 
                            target_attr.out_features,
                            target_attr.bias is not None,
                            self.is_global,
                            self.roast_array,
                            self.ROAST_INIT,
                            self.compression,
                            False,
                            "random",
                            seed)
        if self.is_conv2d(target_attr):
            new_attr = FakeRoastConv2d( target_attr.in_channels,
                            target_attr.out_channels,
                            target_attr.kernel_size,
                            self.is_global,
                            self.roast_array,
                            self.ROAST_INIT,
                            self.compression,
                            target_attr.stride,
                            target_attr.padding,
                            target_attr.dilation,
                            target_attr.groups, 
                            target_attr.bias is not None,
                            target_attr.padding_mode,
                            False,
                            "random",
                            seed)
        if self.is_embedding(target_attr):
            new_attr = FakeRoastEmbedding(target_attr.num_embeddings, 
                            target_attr.embedding_dim,
                            self.is_global, self.roast_array, self.ROAST_INIT, 
                            self.compression, target_attr.padding_idx, 
                            target_attr.max_norm, target_attr.norm_type,
                            target_attr.scale_grad_by_freq, 
                            target_attr.sparse) # missing seed?
    
        return new_attr

    def lambda_init(self, state_dict):
        state_dict['seed'] = state_dict['init_seed'] * 1024
        return state_dict


    def lambda_next(self, state_dict):
        state_dict['init_seed'] = state_dict['init_seed'] + 1
        return state_dict

    def lambda_func(self, state_dict):
        attr = state_dict['target_attr']
        name = state_dict['name']
        state_dict['seed'] = state_dict['seed'] + 1
        new_attr = self.make_roast_module(attr, state_dict['seed'])
        if self.verbose >= DEBUG:
            print(type(attr), new_attr, flush=True)
        if new_attr is not None:
            setattr(state_dict['model'], name, new_attr)

        return state_dict

    def process(self):
        state_dict = {'init_seed' : 1}
        self.run("model", self.model, state_dict)
        return self.model
   
