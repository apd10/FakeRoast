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

test_mapper()
