import torch

from FakeRoast import *
import matplotlib.pyplot as plt
PLOT=False

def testRoastLinearLocal_forward():
    linear = FakeRoastLinear(64, 1024, False, False, None, None, 0.01)
    batch = torch.rand(32, 64)
    output = linear(batch)

    # test the randomness in the idx
    if PLOT:
      plt.hist(linear.WHelper.IDX.view(-1), bins=100)
      plt.show() # should be flat
    else:
      idxs = linear.WHelper.IDX.view(-1).float() / linear.WHelper.wsize
      percentile = torch.arange(0, 1, 0.02)
      observed = torch.quantile(idxs, percentile)
      torch.testing.assert_close(percentile, observed, atol=0.01, rtol=0)

    # test the scale of init
    observed_max, observed_min = torch.max(linear.WHelper() * linear.scale), torch.min(linear.WHelper() * linear.scale)
    expected = sqrt(1. / linear.idim)
    print(observed_max, observed_min, expected)
    assert(torch.abs(observed_max - expected) < 5e-4)
    assert(torch.abs(observed_min + expected) < 5e-4)
      

def testRoastLinearGlobal_forward():
    wsize = 1000
    weight = nn.Parameter(torch.zeros(wsize, dtype=torch.float), requires_grad=True)
    init_scale = 0.0001
    nn.init.uniform_(weight.data, a=-init_scale, b = init_scale)
    linear1 = FakeRoastLinear(64, 128, False, True, weight, init_scale, None)
    linear2 = FakeRoastLinear(128, 256, False, True, weight, init_scale, None)
    batch = torch.rand(32, 64)
    output = linear1(batch)
    output = linear2(output)
    
    # test the randomness in the idx
    if PLOT:
      plt.hist(linear2.WHelper.IDX.view(-1), bins=100)
      plt.show() # should be flat
    else:
      idxs = linear2.WHelper.IDX.view(-1).float() / linear2.WHelper.wsize
      percentile = torch.arange(0, 1, 0.02)
      observed = torch.quantile(idxs, percentile)
      torch.testing.assert_close(percentile, observed, atol=0.01, rtol=0)

    # test the scale of init
    import pdb
    pdb.set_trace()

    observed_max, observed_min = torch.max(linear1.WHelper() * linear1.scale), torch.min(linear1.WHelper() * linear1.scale)
    expected = sqrt(1. / linear1.idim)
    print(observed_max, observed_min, expected)
    assert(torch.abs(observed_max - expected) < 5e-4)
    assert(torch.abs(observed_min + expected) < 5e-4)

    observed_max, observed_min = torch.max(linear2.WHelper() * linear2.scale), torch.min(linear2.WHelper() * linear2.scale)
    expected = sqrt(1. / linear2.idim)
    print(observed_max, observed_min, expected)
    assert(torch.abs(observed_max - expected) < 5e-4)
    assert(torch.abs(observed_min + expected) < 5e-4)
