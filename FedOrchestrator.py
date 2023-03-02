import torch
import torch.nn as nn
import numpy as np
import pdb
from FakeRoast.FakeRoast import *
import pdb


class FedOrchestrator:
    # currently not storing any state. Just a set of gradient/weight communication functions.

    def get_wts_full_from_keys_single(model, keys):
        ''' no need to handle scale here as it is same across all the models '''
        ''' TODO(aditya) handle different device for models '''
        # print("get_wts_full_from_keys_single", flush=True)
        dic = {}
        for n, m in model.named_modules():
            key = n + 'wt'
            if key in keys:
                dic[key] = m.weight.data.clone()
            key = n + 'bs'
            if key in keys:
                dic[key] = m.bias.data.clone()
        return dic

    def get_wts_full_single(model, is_global):
        ''' no need to handle scale here as it is same across all the models '''
        ''' TODO(aditya) handle different device for models '''
        # print("get_wts_full_single", flush=True)
        final_dic = {}
        for n, m in model.named_modules():
            if type(m) in [nn.Linear, nn.Conv2d]:
                final_dic[n+'wt'] = m.weight.data.clone()
                final_dic[n+'bs'] = m.bias.data.clone()
            elif type(m) == nn.Embedding:
                final_dic[n+'wt'] = m.weight.data.clone()
        return final_dic

    def set_wts_full(model, dic):
        ''' no need to handle scale here as it is same across all the models '''
        ''' TODO(aditya) handle different device for models '''
        # print("set_wts_full", flush=True)
        for n, m in model.named_modules():
            key = n + 'wt'
            if key in dic.keys():
                m.weight.data = dic[key].clone()
            key = n + 'bs'
            if key in dic.keys():
                m.bias.data = dic[key].clone()

    def set_wts_roast(model, final_dic, is_global, alpha):
        # print("set_wts_roast", flush=True)
        global_wt = None
        global_ct = None
        for n, m in model.named_modules():
            # print(n, type(m), flush=True)
            if type(m) in [FakeRoastLinear, FakeRoastConv2d, FakeRoastEmbedding]:
                wt, ct = m.WHelper.wt_orig_to_comp(final_dic[n+'wt'])
                if is_global:
                    if global_wt is None:
                        global_wt = wt
                        global_ct = ct
                    else:
                        global_wt = global_wt + wt
                        global_ct = global_ct + ct
                else:
                    m.WHelper.weight.data = alpha * m.WHelper.weight.data + \
                        (1 - alpha) * torch.div(wt, ct+1e-3)

            elif type(m) in [LowRankLinear, LowRankEmbedding]:
                w1, w2 = m.wt_orig_to_comp(final_dic[n+'wt'])
                m.w1.data = alpha * m.w1.data + (1-alpha) * w1
                m.w2.data = alpha * m.w2.data + (1-alpha) * w2

            elif type(m) in [nn.Linear, nn.Conv2d, nn.Embedding]:
                m.weight.data = alpha * m.weight.data + \
                    (1-alpha) * final_dic[n+'wt']
            elif type(m) == FakeRoastLSTM:
                # wt1, ct1 = m.WHelper1.wt_orig_to_comp(final_dic[n+'wt1'])
                wt, ct = m.WHelper2.wt_orig_to_comp(final_dic[n+'wt'])
                assert is_global == False
                # m.WHelper1.weight.data = alpha * m.WHelper1.weight.data + (1 - alpha) * torch.div(wt1, ct1+1e-3)
                m.WHelper2.weight.data = alpha * m.WHelper2.weight.data + \
                    (1 - alpha) * torch.div(wt, ct+1e-3)
            else:
                if n not in ['', 'fc', 'fc.1', 'fc.3'] and (not n.endswith('WHelper')) and (not type(m) in [nn.Dropout]):
                    print("[set]NOT FOUND MODULE __ CHECK :", n)

            if type(m) in [FakeRoastLinear, FakeRoastConv2d, FakeRoastLSTM, nn.Linear, nn.Conv2d, LowRankLinear]:
                m.bias.data = alpha * m.bias.data + \
                    (1 - alpha) * final_dic[n+'bs']

        # TODO(aditya) global behavior is untested
        if is_global:
            for n, m in model.named_modules():
                if type(m) in [FakeRoastLinear, FakeRoastConv2d, FakeRoastEmbedding]:
                    m.WHelper.weight.data = alpha * m.WHelper.weight.data + \
                        (1 - alpha) * torch.div(global_wt, global_ct+1e-3)
                    # update only once
                    break

    def set_wts(models, final_dic, is_global, is_server, alpha=0):
        # print("set_wts", flush=True)
        for i in range(len(models)):
            model = models[i]
            if is_server:
                FedOrchestrator.set_wts_full(model, final_dic)
            else:
                FedOrchestrator.set_wts_roast(
                    model, final_dic, is_global, alpha)

    def get_wts_full(models, weights):
        ''' no need to handle scale here as it is same across all the models '''
        ''' TODO(aditya) handle different device for models '''
        # print("get_wts_full", flush=True)
        dics = []
        for i in range(len(models)):
            model = models[i]
            dics.append({})
            weight = weights[i]
            for n, m in model.named_modules():
                if type(m) in [nn.Linear, nn.Conv2d]:
                    dics[i][n+'wt'] = m.weight.data.clone() * weight
                    dics[i][n+'bs'] = m.bias.data.clone() * weight
                elif type(m) == nn.Embedding:
                    dics[i][n+'wt'] = m.weight.data.clone() * weight

        # wt avg
        final_dic = {}
        for k in dics[0].keys():
            final_dic[k] = torch.clone(dics[0][k])
            for dic in dics[1:]:
                final_dic[k] = final_dic[k] + dic[k]
            # final_dic[k] = (final_dic[k] / len(dics)) weighted average using weights
        return final_dic

    def get_wts_roast(models, is_global, weights):
        ''' no need to handle scale here as it is same across all the models '''
        ''' TODO(aditya) handle different device for models '''
        # print("get_wts_roast", flush=True)
        dics = []
        for i in range(len(models)):
            model = models[i]
            weight = weights[i]
            dics.append({})
            for n, m in model.named_modules():
                # print(n, type(m), flush=True)
                if type(m) in [FakeRoastLinear, FakeRoastConv2d]:
                    dics[i][n+'wt'] = m.WHelper.wt_comp_to_orig(
                        m.WHelper.weight.data) * weight
                    dics[i][n+'bs'] = m.bias.data * weight
                elif type(m) in [LowRankLinear, LowRankEmbedding]:
                    dics[i][n +
                            'wt'] = m.wt_comp_to_orig(m.w1.data, m.w2.data) * weight
                    if type(m) in [LowRankLinear]:
                        dics[i][n+'bs'] = m.bias.data * weight
                elif type(m) == FakeRoastEmbedding:
                    dics[i][n+'wt'] = m.WHelper.wt_comp_to_orig(
                        m.WHelper.weight.data) * weight
                elif type(m) in [nn.Linear, nn.Conv2d]:
                    dics[i][n+'wt'] = m.weight.data.clone() * weight
                    dics[i][n+'bs'] = m.bias.data.clone() * weight
                elif type(m) == nn.Embedding:
                    dics[i][n+'wt'] = m.weight.data.clone() * weight
                elif type(m) == FakeRoastLSTM:
                    # dics[i][n+'wt1'] = m.WHelper1.wt_comp_to_orig(m.WHelper1.weight.data) * weight
                    dics[i][n+'wt'] = m.WHelper2.wt_comp_to_orig(
                        m.WHelper2.weight.data) * weight
                    dics[i][n+'bs'] = m.bias.data * weight
                else:
                    if n not in ['', 'fc', 'fc.1', 'fc.3'] and (not n.endswith('WHelper')) and (not type(m) in [nn.Dropout]):
                        print("[set]NOT FOUND MODULE __ CHECK :", n)

        # wt avg
        final_dic = {}
        for k in dics[0].keys():
            final_dic[k] = torch.clone(dics[0][k])
            for dic in dics[1:]:
                final_dic[k] = final_dic[k] + dic[k]
            final_dic[k] = final_dic[k]  # / len(dics)
        return final_dic

    def get_wts_roast_median(models, is_global, weights):
        ''' no need to handle scale here as it is same across all the models '''
        ''' TODO(aditya) handle different device for models '''
        # print("get_wts_roast_median", flush=True)
        dics = []
        for i in range(len(models)):
            model = models[i]
            dics.append({})
            weight = weights[i]
            for n, m in model.named_modules():
                if type(m) in [FakeRoastLinear, FakeRoastConv2d]:
                    dics[i][n+'wt'] = m.WHelper.wt_comp_to_orig(
                        m.WHelper.weight.data) * weight
                    dics[i][n+'bs'] = m.bias.data * weight
                elif type(m) == FakeRoastEmbedding:
                    dics[i][n+'wt'] = m.WHelper.wt_comp_to_orig(
                        m.WHelper.weight.data) * weight
                else:
                    raise NotImplementedError

        # wt avg
        final_dic = {}
        for k in dics[0].keys():
            ll = []
            for dic in dics:
                ll.append(dic[k])
            ll = torch.stack(ll)
            final_dic[k], _ = torch.median(ll, dim=0)
        return final_dic

    def get_wts(models, is_global, is_server, weights):
        if is_server:
            return FedOrchestrator.get_wts_full(models)
        else:
            return FedOrchestrator.get_wts_roast(models, is_global, weights)

    def merge_wt_normalize(models, is_global, alpha=0):
        ''' no need to handle scale here as it is same across all the models '''
        ''' TODO(aditya) handle different device for models '''
        # print("merge_wt_normalize", flush=True)
        dics = []
        for i in range(len(models)):
            model = models[i]
            dics.append({})
            for n, m in model.named_modules():
                if type(m) in [FakeRoastLinear, FakeRoastConv2d]:
                    dics[i][n +
                            'wt'] = m.WHelper.wt_comp_to_orig(m.WHelper.weight.data)
                    dics[i][n+'bs'] = m.bias.data
                elif type(m) == FakeRoastEmbedding:
                    dics[i][n +
                            'wt'] = m.WHelper.wt_comp_to_orig(m.WHelper.weight.data)

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
                if type(m) in [FakeRoastLinear, FakeRoastConv2d, FakeRoastEmbedding]:
                    wt, ct = m.WHelper.wt_orig_to_comp(final_dic[n+'wt'])
                    if is_global:
                        if global_wt is None:
                            global_wt = wt
                            global_ct = ct
                        else:
                            global_wt = global_wt + wt
                            global_ct = global_ct + ct
                    else:
                        m.WHelper.weight.data = alpha * m.WHelper.weight.data + \
                            (1 - alpha) * torch.div(wt, ct+1e-3)

                if type(m) in [FakeRoastLinear, FakeRoastConv2d]:
                    m.bias.data = alpha * m.bias.data + \
                        (1 - alpha) * final_dic[n+'bs']

            if is_global:
                for n, m in model.named_modules():
                    if type(m) in [FakeRoastLinear, FakeRoastConv2d, FakeRoastEmbedding]:
                        m.WHelper.weight.data = alpha * m.WHelper.weight.data + \
                            (1 - alpha) * torch.div(global_wt, global_ct+1e-3)
                        # update only once
                        break
