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
import copy
try:
    from .FakeRoast import *
except:
    from FakeRoast import *

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
        self.FAKELINEAR = [FakeRoastLinear]
        self.FAKECONV2D = [FakeRoastConv2d]
        self.FAKEEMBEDDING = [FakeRoastEmbedding]
        self.CONV2D = [torch.nn.Conv2d, torch.nn.modules.Conv2d]
        self.EMBEDDING = [torch.nn.Embedding, torch.nn.modules.Embedding]
        self.module_limit_size=module_limit_size
        self.verbose=verbose


    def is_fakeroasted(self, attr):
        return type(attr) in (self.FAKELINEAR + self.FAKECONV2D + self.FAKEEMBEDDING)

    def is_linear(self, attr):
        return type(attr) in self.LINEAR

    def is_conv2d(self, attr):
        return type(attr) in self.CONV2D

    def is_embedding(self, attr):
        return type(attr) in self.EMBEDDING

    def is_roast_linear(self, attr):
        return type(attr) in self.FAKELINEAR

    def is_roast_conv2d(self, attr):
        return type(attr) in self.FAKECONV2D

    def is_roast_embedding(self, attr):
        return type(attr) in self.FAKEEMBEDDING

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
    
    def get_in_channel_dim(self, attr):
        assert(self.roastable(attr))
        idc = id(attr.weight)
        c = np.prod(attr.weight.shape[1:])
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
                  idin, ins = self.get_in_channel_dim(attr)
                  state_dict['compressable'][idc] = c
                  state_dict['reducible'][idin] = ins

        return state_dict

    def process(self):
        state_dict = {"compressable" : {}, "all" : {}, "reducible" : {}}
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

    def __init__(self, model, roast_global, sparsity, module_limit_size=None, verbose=NONE, init_std=0.04, mapper_args=None):
        ModelParser.__init__(self)
        Roastable.__init__(self, module_limit_size=module_limit_size, verbose=verbose)
      
        self.verbose = verbose

        self.ROAST_INIT = init_std
        self.model = model
        self.is_global = roast_global
        self.compression = sparsity

        parameter_finder = ModelRoastableParameters(model, module_limit_size=module_limit_size)
        pf = parameter_finder.process()
        roastable_params, total_params = pf['roastable'], pf['all']

        if mapper_args is not None:

            self.global_flop_reduction = 1 - mapper_args['comp_reduction_rate']
            self.total_flop_reducible = sum(pf['reducible'].values())
            self.reduction_factor = self.global_flop_reduction * (self.total_flop_reducible / len(pf['reducible']))
            self.adjust_layer_flop_reduction(pf['reducible'])

            


        if self.verbose >= INFO:
            print("Roastable params: {}/{}".format(roastable_params, total_params))

        if self.is_global:
            ''' need to compute the max params: sparsity is applied to roastable parameters '''
            max_params = int(sparsity * roastable_params)
            #self.roast_array = torch.nn.Parameter(torch.FloatTensor(max_params).uniform_(-self.ROAST_INIT, self.ROAST_INIT))
            self.roast_array = torch.nn.Parameter(torch.FloatTensor(max_params).normal_(std=self.ROAST_INIT))
        else:
            self.roast_array = None

        self.original_total_params = total_params
        self.original_roastable_params = roastable_params
        self.global_offset = 0
        self.mapper_args = mapper_args

    def make_roast_module(self, target_attr, seed):
        if not self.roastable(target_attr):
              return None
        new_attr = None

        if self.mapper_args is not None:
            mapper_args = copy.deepcopy(self.mapper_args)
            mapper_args["original_offset"] = self.global_offset
            mapper_args["seed"] = seed
            self.layer_flop_reduction = self.layers_flop_reduction[id(target_attr.weight)]
            mapper_args["comp_reduction_rate"] = 1 - self.layer_flop_reduction 
        else:
            mapper_args = None

        if self.is_linear(target_attr):
            new_attr = FakeRoastLinear(target_attr.in_features, 
                            target_attr.out_features,
                            target_attr.bias is not None,
                            self.is_global,
                            self.roast_array,
                            self.ROAST_INIT,
                            self.compression,
                            False,
                            "mapper" if (mapper_args is not None) else "random",
                            seed,
                            req_scale = torch.std(target_attr.weight).item(),
                            mapper_args = mapper_args)
            self.global_offset = self.global_offset + target_attr.weight.numel()

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
                            "mapper" if (mapper_args is not None) else "random",
                            seed,
                            req_scale = torch.std(target_attr.weight).item(),
                            mapper_args = mapper_args)
            self.global_offset = self.global_offset + target_attr.weight.numel()
        if self.is_embedding(target_attr):
            new_attr = FakeRoastEmbedding(target_attr.num_embeddings, 
                            target_attr.embedding_dim,
                            self.is_global, self.roast_array, self.ROAST_INIT, 
                            self.compression, target_attr.padding_idx, 
                            target_attr.max_norm, target_attr.norm_type,
                            target_attr.scale_grad_by_freq, 
                            target_attr.sparse,
                            matrix_mode= "mapper" if (mapper_args is not None) else "random",
                            req_scale = torch.std(target_attr.weight).item(),
                            mapper_args = mapper_args) # missing seed?
            self.global_offset = self.global_offset + target_attr.weight.numel()
    
        return new_attr
    
    
    def adjust_layer_flop_reduction(self, reducibles):

        self.layers_flop_reduction = {}
        
        for layer_id, layer_reducible in reducibles.items():
            self.layer_flop_reduction = self.reduction_factor / layer_reducible
            self.layers_flop_reduction[layer_id] = self.layer_flop_reduction 
        
        while not all(0 <= value <= 1 for value in self.layers_flop_reduction.values()):
            residue_sum = 0
            for layer_id, layer_flop_reduction in self.layers_flop_reduction.items():
                if layer_flop_reduction > 1:
                    residue_sum += (layer_flop_reduction - 1) * reducibles[layer_id]
                    self.layers_flop_reduction[layer_id] = 1
            compensation_factor = residue_sum / sum([reducibles[item] for item, value in self.layers_flop_reduction.items() if value < 1])
            for layer_id, layer_flop_reduction in self.layers_flop_reduction.items():
                if layer_flop_reduction < 1:
                    self.layers_flop_reduction[layer_id] += compensation_factor 
                            

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
        if self.is_global:
            self.model.roast_array = self.roast_array
        else:
            self.model.roast_array = None
        return self.model
   

class ModelRoasterGradScaler(ModelRoaster):
    def __init__(self, model, roast_global, sparsity, module_limit_size=None, verbose=NONE, init_std=0.04, scaler_mode="v1", mapper_args=None):
        super(ModelRoasterGradScaler, self).__init__(model, roast_global, sparsity, module_limit_size=None, verbose=NONE, init_std=init_std,
                                                     mapper_args=mapper_args)
        assert(roast_global) # this should be defined only for roast_global
        self.scaler_mode = scaler_mode
        self.count = torch.zeros_like(self.roast_array)
        self.aggregate_scale = torch.zeros_like(self.roast_array)
        self.aggregate2_scale = torch.zeros_like(self.roast_array)


    def lambda_func(self, state_dict):
        attr = state_dict['target_attr']
        name = state_dict['name']
        state_dict['seed'] = state_dict['seed'] + 1
        new_attr = self.make_roast_module(attr, state_dict['seed'])
        if self.verbose >= DEBUG:
            print(type(attr), new_attr, flush=True)
        if new_attr is not None:
            setattr(state_dict['model'], name, new_attr)
            weight = new_attr.WHelper()
            _, count = new_attr.WHelper.wt_orig_to_comp(weight)
            self.count += count
            # note that this is sum of scale factors without sign G
            self.aggregate_scale += count * new_attr.scale 
            self.aggregate2_scale += count * new_attr.scale**2
        return state_dict

    def compute_roast_grad_scale_v1(self):
      return torch.square(self.aggregate_scale) / (1e-3 + self.count)

    def compute_roast_grad_scale_v2(self):
      return self.aggregate_scale

    def compute_roast_grad_scale_v3(self):
      return (self.aggregate_scale) / (1e-3 + self.count)

    def compute_roast_grad_scale_v4(self):
      return torch.sqrt(self.aggregate2_scale)

    def compute_roast_grad_scale_v5(self):
      return self.aggregate2_scale

    def compute_roast_grad_scale_none(self):
      return torch.ones_like(self.aggregate_scale)

    def compute_roast_grad_scale(self):
      if self.scaler_mode == "v1":
          return self.compute_roast_grad_scale_v1()
      if self.scaler_mode == "v2":
          return self.compute_roast_grad_scale_v2()
      if self.scaler_mode == "v3":
          return self.compute_roast_grad_scale_v3()
      if self.scaler_mode == "v4":
          return self.compute_roast_grad_scale_v4()
      if self.scaler_mode == "v5":
          return self.compute_roast_grad_scale_v5()
      if self.scaler_mode == "none":
          return self.compute_roast_grad_scale_none()
      raise NotImplementedError

    '''
    def compute_roast_grad_scale(self):
      #return torch.square(self.aggregate_scale) / (1e-3 + self.count)
      return (self.aggregate_scale) / (1e-3 + self.count)
      #return self.aggregate_scale
    '''  
    def process(self):
        super().process()
        self.roast_array._roast_grad_scaler = self.compute_roast_grad_scale()
        return self.model


class RoastGradScaler:
    def __init__(self):
        pass

    def scale_step(self, model):
        if not('roast_array' in dir(model)):
            return

        for p in model.parameters():
            if (p.requires_grad) and (p.grad is not None) and '_roast_grad_scaler' in dir(p):
                p._roast_grad_scaler = p._roast_grad_scaler.to(p.device)
                p.grad = p.grad / (1e-6+p._roast_grad_scaler)



class RoastToFullModel(ModelParser, Roastable):
    def __init__(self, roast_model):
          ModelParser.__init__(self)
          Roastable.__init__(self)
          self.model = roast_model

    def change_to_full_module(self, target_attr):
        if not self.is_fakeroasted(target_attr):
            return None
        if self.is_roast_linear(target_attr):
            new_attr = nn.Linear(target_attr.idim, 
                            target_attr.odim,
                            target_attr.bias is not None)
            new_attr.weight.data[:,:] = target_attr.WHelper() * target_attr.scale
            if target_attr.bias is not None:
                new_attr.bias.data[:] = target_attr.bias

        if self.is_roast_conv2d(target_attr):
            new_attr = nn.Conv2d(target_attr.in_channels,
                    target_attr.out_channels,
                    target_attr.kernel_size,
                    stride=target_attr.stride,
                    padding=target_attr.padding,
                    dilation=target_attr.dilation,
                    groups=target_attr.groups, 
                    bias=target_attr.bias is not None,
                    padding_mode=target_attr.padding_mode)
    
            new_attr.weight.data[:,:,:,:] = target_attr.WHelper() * target_attr.scale
            if target_attr.bias is not None:
                new_attr.bias.data[:] = target_attr.bias


        if self.is_roast_embedding(target_attr):
            new_attr = nn.Embedding(target_attr.num_embeddings, 
                            target_attr.embedding_dim,
                            max_norm = target_attr.max_norm,
                            norm_type = target_attr.norm_type,
                            scale_grad_by_freq = target_attr.scale_grad_by_freq, 
                            sparse = target_attr.sparse) # missing seed?
            
            new_attr.weight.data[:,:] = target_attr.WHelper() * target_attr.scale

        return new_attr


    def lambda_func(self, state_dict):
        attr = state_dict['target_attr']
        name = state_dict['name']
        new_attr = self.change_to_full_module(attr)
        if self.verbose >= DEBUG:
            print(type(attr), new_attr, flush=True)
        if new_attr is not None:
            setattr(state_dict['model'], name, new_attr)
        return state_dict

    def process(self):
        state_dict = {}
        self.run("model", self.model, state_dict)
        if 'roast_array' in dir(self.model):
            del self.model.roast_array 
        return self.model
