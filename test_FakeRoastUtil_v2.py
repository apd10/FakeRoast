import torchvision
from FakeRoastUtil_v2 import *


def test_1(sparsity=0.5):
    model = torchvision.models.AlexNet()
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


