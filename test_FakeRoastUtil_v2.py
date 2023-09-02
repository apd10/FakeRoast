import torchvision
from FakeRoastUtil_v2 import *
import pdb


def test1(sparsity=0.5):
    model = torchvision.models.AlexNet()
    print(model)
    bef = get_module_params(model)
    pf = ModelRoastableParameters(model, module_limit_size=25000)
    s = pf.process()
    roastable = s['roastable']
    total = s['all']
    fixed = total - roastable
    
    roaster = ModelRoaster(model, False, sparsity, verbose=NONE, module_limit_size=25000)
    model = roaster.process()
    print(model)
    af = get_module_params(model)
    print(int(roastable * sparsity), (af-fixed))
    assert(abs( int(roastable * sparsity) - (af - fixed)) < 5)

def test2(sparsity=0.5):
    model = torchvision.models.AlexNet()
    import pdb
    pdb.set_trace()
    print(model)
    bef = get_module_params(model)
    pf = ModelRoastableParameters(model)
    s = pf.process()
    roastable = s['roastable']
    total = s['all']
    fixed = total - roastable
    
    roaster = ModelRoaster(model, False, sparsity, verbose=NONE)
    model = roaster.process()
    print(model)
    af = get_module_params(model)
    print(int(roastable * sparsity), (af-fixed))
    assert(abs( int(roastable * sparsity) - (af - fixed)) < 5)


def test3(sparsity=0.5):
    model = torchvision.models.AlexNet()
    print(model)
    bef = get_module_params(model)
    pf = ModelRoastableParameters(model)
    s = pf.process()
    roastable = s['roastable']
    total = s['all']
    fixed = total - roastable
    
    roaster = ModelRoasterGradScaler(model, True, sparsity, verbose=NONE)
    model = roaster.process()
    print(model)


def test_mapper(sparsity=0.5):
    model = torchvision.models.AlexNet()
    mapper_args = { "mapper":"pareto", "hasher" : "uhash", "block_k" : 16, "block_n" : 16, "block": 8, "seed" : 1011 }
    roaster = ModelRoasterGradScaler(model, True, sparsity, verbose=NONE, mapper_args=mapper_args)
    model = roaster.process()
    print(model)


def test_change(sparsity=0.5):
    model = torchvision.models.AlexNet()
    mapper_args = { "mapper":"pareto", "hasher" : "uhash", "block_k" : 16, "block_n" : 16, "block": 8, "seed" : 1011 }
    roaster = ModelRoasterGradScaler(model, True, sparsity, verbose=NONE, mapper_args=mapper_args)
    model = roaster.process()
    model.features[6].bias.data[:] = 1
    model.classifier[1].bias.data[:] = 1

    backup = copy.deepcopy(model)
    hyderator = RoastToFullModel(model)
    full_model = hyderator.process()
    print(full_model)
    backup_dict = dict(backup.named_parameters())
    backup_modules_dict = dict(backup.named_modules())
    for n,w in full_model.named_parameters():
        print( " ===== ", n, "======")
        print("full", torch.sum(w))
        if 'weight' in n and  n.replace('weight', 'WHelper.IDX') in backup_dict.keys():
            print("roast", torch.sum(backup_modules_dict[n.replace('.weight', '')].WHelper()))
        else:
            print("roast", torch.sum(backup_dict[n]))

test_change()
