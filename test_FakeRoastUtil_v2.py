import torchvision
from FakeRoastUtil_v2 import *
import pdb
import matplotlib.pyplot as plt


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
    mapper_args = { "mapper":"roast_comp", "hasher" : "uhash", "block_k" : 5, "block_n" : 5, "block": 5, "seed" : 1011, "block_k_small": 2}
    roaster = ModelRoasterGradScaler(model, True, sparsity, verbose=NONE, mapper_args=mapper_args)
    roaster.roast_array.data[:] = torch.arange(roaster.roast_array.numel())
    model = roaster.process()
    print(model)
    print("classifier.6", model.classifier[6].WHelper().shape)
    plt.imshow(torch.abs(model.classifier[6].WHelper()).long()[:50,:50], cmap='hot', interpolation='nearest')
    print(torch.abs(model.classifier[6].WHelper()).long()[:50,:50])
    plt.show()


def test_change(sparsity=0.5):
    model = torchvision.models.AlexNet()
    mapper_args = { "mapper":"pareto", "hasher" : "uhash", "block_k" : 32, "block_n" : 32, "block": 32, "seed" : 1011, "block_k_small": 8 }
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

test_mapper()
