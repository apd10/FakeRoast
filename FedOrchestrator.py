import torch
import torch.nn as nn
import numpy as np
import pdb
from FakeRoast import *

class FedOrchestrator:
    # currently not storing any state. Just a set of gradient/weight communication functions. 
    
    def merge_wt_normalize(models, is_global, alpha = 0):
        ''' no need to handle scale here as it is same across all the models '''
        ''' TODO(aditya) handle different device for models '''
        dics = []
        for i in range(len(models)):
            model = models[i]
            dics.append({})
            for n, m in model.named_modules():
                if type(m) in [FakeRoastLinear, FakeRoastConv2d] :
                    dics[i][n+'wt'] = m.WHelper.wt_comp_to_orig(m.WHelper.weight.data)
                    dics[i][n+'bs'] = m.bias.data
                elif type(m) == FakeRoastEmbedding:
                    dics[i][n+'wt'] = m.WHelper.wt_comp_to_orig(m.WHelper.weight.data)
        
        # wt avg
        final_dic = {}
        for k in dics[0].keys():
            final_dic[k] = torch.clone(dics[0][k])
            for dic in dics[1:]:
                final_dic[k] = final_dic[k] + dic[k]
            final_dic[k] = final_dic[k] / len(dics)
        
                

        for i in range(len(models)):
            model = models[i]
            global_wt = None
            global_ct = None
            for n, m in model.named_modules():
                if type(m) in [FakeRoastLinear, FakeRoastConv2d, FakeRoastEmbedding] :
                    wt, ct = m.WHelper.wt_orig_to_comp(final_dic[n+'wt'])
                    if is_global:
                        if global_wt is None:
                            global_wt = wt
                            global_ct = ct
                        else:
                            global_wt = global_wt + wt
                            global_ct = global_ct + ct
                    else:    
                        m.WHelper.weight.data = alpha * m.WHelper.weight.data + (1 - alpha) * torch.div(wt, ct+1e-3)
                
                if type(m) in [FakeRoastLinear, FakeRoastConv2d] :
                    m.bias.data = alpha * m.bias.data + (1 - alpha) * final_dic[n+'bs']

            if is_global:
                for n, m in model.named_modules():
                    if type(m) in [FakeRoastLinear, FakeRoastConv2d, FakeRoastEmbedding] :
                        m.WHelper.weight.data = alpha * m.WHelper.weight.data + (1 - alpha) * torch.div(global_wt, global_ct+1e-3)
                        # update only once
                        break

