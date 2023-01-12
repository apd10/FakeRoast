import torch
import pdb
from FakeRoast.FakeRoast import *
import matplotlib.pyplot as plt
PLOT = False


def testRoastLinearLocal_forward():
    linear = FakeRoastLinear(64, 1024, False, False, None, None, 0.01)
    batch = torch.rand(32, 64)
    output = linear(batch)

    # test the randomness in the idx
    if PLOT:
        plt.hist(linear.WHelper.IDX.view(-1), bins=100)
        plt.show()  # should be flat
    else:
        idxs = linear.WHelper.IDX.view(-1).float() / linear.WHelper.wsize
        percentile = torch.arange(0, 1, 0.02)
        observed = torch.quantile(idxs, percentile)
        torch.testing.assert_close(percentile, observed, atol=0.01, rtol=0)

    # test the scale of init
    observed_max, observed_min = torch.max(
        linear.WHelper() * linear.scale), torch.min(linear.WHelper() * linear.scale)
    expected = sqrt(1. / linear.idim)
    print(observed_max, observed_min, expected)
    assert (torch.abs(observed_max - expected) < 5e-4)
    assert (torch.abs(observed_min + expected) < 5e-4)


def testRoastLinearGlobal_forward():
    wsize = 1000
    weight = nn.Parameter(torch.zeros(
        wsize, dtype=torch.float), requires_grad=True)
    init_scale = 0.0001
    nn.init.uniform_(weight.data, a=-init_scale, b=init_scale)
    linear1 = FakeRoastLinear(64, 128, False, True, weight, init_scale, None)
    linear2 = FakeRoastLinear(128, 256, False, True, weight, init_scale, None)
    batch = torch.rand(32, 64)
    output = linear1(batch)
    output = linear2(output)

    # test the randomness in the idx
    if PLOT:
        plt.hist(linear2.WHelper.IDX.view(-1), bins=100)
        plt.show()  # should be flat
    else:
        idxs = linear2.WHelper.IDX.view(-1).float() / linear2.WHelper.wsize
        percentile = torch.arange(0, 1, 0.02)
        observed = torch.quantile(idxs, percentile)
        torch.testing.assert_close(percentile, observed, atol=0.01, rtol=0)

    # test the scale of init
    observed_max, observed_min = torch.max(
        linear1.WHelper() * linear1.scale), torch.min(linear1.WHelper() * linear1.scale)
    expected = sqrt(1. / linear1.idim)
    print(observed_max, observed_min, expected)
    assert (torch.abs(observed_max - expected) < 5e-4)
    assert (torch.abs(observed_min + expected) < 5e-4)

    observed_max, observed_min = torch.max(
        linear2.WHelper() * linear2.scale), torch.min(linear2.WHelper() * linear2.scale)
    expected = sqrt(1. / linear2.idim)
    print(observed_max, observed_min, expected)
    assert (torch.abs(observed_max - expected) < 5e-4)
    assert (torch.abs(observed_min + expected) < 5e-4)


def testRoastConvLocal_forward():
    conv2d = FakeRoastConv2d(32, 1024, 4, False, None, None, 0.01)
    batch = torch.rand(128, 32, 32, 32)
    output = conv2d(batch)

    # test the randomness in the idx

    if PLOT:
        plt.hist(conv2d.WHelper.IDX.view(-1), bins=100)
        plt.show()  # should be flat
    else:
        idxs = conv2d.WHelper.IDX.view(-1).float() / conv2d.WHelper.wsize
        percentile = torch.arange(0, 1, 0.02)
        observed = torch.quantile(idxs, percentile)
        torch.testing.assert_close(percentile, observed, atol=0.02, rtol=0)
    # test the scale of init
    observed_max, observed_min = torch.max(
        conv2d.WHelper() * conv2d.scale), torch.min(conv2d.WHelper() * conv2d.scale)
    expected = sqrt(1.0 * conv2d.groups /
                    (conv2d.in_channels * np.prod(conv2d.kernel_size)))
    print(observed_max, observed_min, expected)
    assert (torch.abs(observed_max - expected) < 5e-4)
    assert (torch.abs(observed_min + expected) < 5e-4)


def testRoastConvGlobal_forward():
    wsize = 1000
    weight = nn.Parameter(torch.zeros(
        wsize, dtype=torch.float), requires_grad=True)
    init_scale = 0.0001
    nn.init.uniform_(weight.data, a=-init_scale, b=init_scale)
    conv2d1 = FakeRoastConv2d(32, 32, 4, True, weight, init_scale, None)
    conv2d2 = FakeRoastConv2d(32, 32, 4, True, weight, init_scale, None)
    batch = torch.rand(128, 32, 32, 32)
    output = conv2d1(batch)
    output = conv2d2(output)

    # test the randomness in the idx
    if PLOT:
        plt.hist(conv2d2.WHelper.IDX.view(-1), bins=100)
        plt.show()  # should be flat
    else:
        idxs = conv2d2.WHelper.IDX.view(-1).float() / conv2d2.WHelper.wsize
        percentile = torch.arange(0, 1, 0.02)
        observed = torch.quantile(idxs, percentile)
        torch.testing.assert_close(percentile, observed, atol=0.02, rtol=0)

    # test the scale of init
    observed_max, observed_min = torch.max(
        conv2d1.WHelper() * conv2d1.scale), torch.min(conv2d1.WHelper() * conv2d1.scale)
    expected = sqrt(1.0 * conv2d1.groups /
                    (conv2d1.in_channels * np.prod(conv2d1.kernel_size)))
    print(observed_max, observed_min, expected)
    assert (torch.abs(observed_max - expected) < 5e-4)
    assert (torch.abs(observed_min + expected) < 5e-4)

    observed_max, observed_min = torch.max(
        conv2d2.WHelper() * conv2d2.scale), torch.min(conv2d2.WHelper() * conv2d2.scale)
    expected = sqrt(1.0 * conv2d2.groups /
                    (conv2d2.in_channels * np.prod(conv2d2.kernel_size)))
    print(observed_max, observed_min, expected)
    assert (torch.abs(observed_max - expected) < 5e-4)
    assert (torch.abs(observed_min + expected) < 5e-4)


def testRoastEmbeddingLocal_forward():
    emb = FakeRoastEmbedding(10000, 32, False, None, None, 0.01)
    batch = torch.randint(0, 10000, (32,))
    output = emb(batch)

    # test the randomness in the idx

    if PLOT:
        plt.hist(emb.WHelper.IDX.view(-1), bins=100)
        plt.show()  # should be flat
    else:
        idxs = emb.WHelper.IDX.view(-1).float() / emb.WHelper.wsize
        percentile = torch.arange(0, 1, 0.02)
        observed = torch.quantile(idxs, percentile)
        torch.testing.assert_close(percentile, observed, atol=0.02, rtol=0)

    # test the scale of init
    observed_max, observed_min = torch.max(
        emb.WHelper() * emb.scale), torch.min(emb.WHelper() * emb.scale)
    expected = sqrt(1.0 / emb.num_embeddings)
    print(observed_max, observed_min, expected)
    assert (torch.abs(observed_max - expected) < 5e-4)
    assert (torch.abs(observed_min + expected) < 5e-4)


def testRoastEmbeddingGlobal_forward():
    wsize = 1000
    weight = nn.Parameter(torch.zeros(
        wsize, dtype=torch.float), requires_grad=True)
    init_scale = 0.0001
    nn.init.uniform_(weight.data, a=-init_scale, b=init_scale)
    emb1 = FakeRoastEmbedding(10000, 32, True, weight, init_scale, None)
    emb2 = FakeRoastEmbedding(10000, 32, True, weight, init_scale, None)
    batch = torch.randint(0, 10000, (32,))
    output = emb1(batch)
    output = emb2(batch)
    # test the randomness in the idx
    if PLOT:
        plt.hist(emb2.WHelper.IDX.view(-1), bins=100)
        plt.show()  # should be flat
    else:
        idxs = emb2.WHelper.IDX.view(-1).float() / emb2.WHelper.wsize
        percentile = torch.arange(0, 1, 0.02)
        observed = torch.quantile(idxs, percentile)
        torch.testing.assert_close(percentile, observed, atol=0.02, rtol=0)

    # test the scale of init
    observed_max, observed_min = torch.max(
        emb1.WHelper() * emb1.scale), torch.min(emb1.WHelper() * emb1.scale)
    expected = sqrt(1.0 / emb1.num_embeddings)
    print(observed_max, observed_min, expected)
    assert (torch.abs(observed_max - expected) < 5e-4)
    assert (torch.abs(observed_min + expected) < 5e-4)

    observed_max, observed_min = torch.max(
        emb2.WHelper() * emb2.scale), torch.min(emb2.WHelper() * emb2.scale)
    expected = sqrt(1.0 / emb2.num_embeddings)
    print(observed_max, observed_min, expected)
    assert (torch.abs(observed_max - expected) < 5e-4)
    assert (torch.abs(observed_min + expected) < 5e-4)


def testRoastLSTMLocal_forward():
    lstm = FakeRoastLSTM(input_dim=4, n_hidden=1024,
                         is_global=False, init_scale=None, compression=0.5)
    batch = torch.rand(128, 1024, 4)
    output = lstm(batch)

    # test the randomness in the idx
    if PLOT:
        plt.hist(lstm.WHelper1.IDX.view(-1), bins=100)
        plt.show()  # should be flat
    else:
        idxs = lstm.WHelper1.IDX.view(-1).float() / lstm.WHelper1.wsize
        percentile = torch.arange(0, 1, 0.02)
        observed = torch.quantile(idxs, percentile)
        torch.testing.assert_close(percentile, observed, atol=0.02, rtol=0)

    # test the scale of init
    observed_max, observed_min = torch.max(
        lstm.WHelper1() * lstm.scale), torch.min(lstm.WHelper1() * lstm.scale)
    expected = sqrt(1. / lstm.n_hidden)
    print(observed_max, observed_min, expected)
    assert (torch.abs(observed_max - expected) < 5e-4)
    assert (torch.abs(observed_min + expected) < 5e-4)


def testRoastLSTMGlobal_forward():
    wsize = 1000
    weight = nn.Parameter(torch.zeros(
        wsize, dtype=torch.float32), requires_grad=True)
    init_scale = 0.0001
    nn.init.uniform_(weight.data, a=-init_scale, b=init_scale)
    lstm1 = FakeRoastLSTM(input_dim=4, n_hidden=1024,
                          is_global=True, init_scale=init_scale, compression=0.5, weight=weight)
    lstm2 = FakeRoastLSTM(input_dim=4, n_hidden=1024,
                          is_global=True, init_scale=init_scale, compression=0.5, weight=weight)
    batch = torch.rand(128, 1024, 4)
    output = lstm1(batch)
    output = lstm2(batch)

    # test the randomness in the idx
    if PLOT:
        plt.hist(lstm2.WHelper1.IDX.view(-1), bins=100)
        plt.show()  # should be flat
    else:
        idxs = lstm2.WHelper1.IDX.view(-1).float() / lstm2.WHelper1.wsize
        percentile = torch.arange(0, 1, 0.02)
        observed = torch.quantile(idxs, percentile)
        torch.testing.assert_close(percentile, observed, atol=0.02, rtol=0)

    # test the scale of init
    observed_max, observed_min = torch.max(
        lstm1.WHelper1() * lstm1.scale), torch.min(lstm1.WHelper1() * lstm1.scale)
    expected = sqrt(1. / lstm1.n_hidden)
    print(observed_max, observed_min, expected)
    assert (torch.abs(observed_max - expected) < 5e-4)
    assert (torch.abs(observed_min + expected) < 5e-4)

    observed_max, observed_min = torch.max(
        lstm2.WHelper1() * lstm2.scale), torch.min(lstm2.WHelper1() * lstm2.scale)
    expected = sqrt(1. / lstm2.n_hidden)
    print(observed_max, observed_min, expected)
    assert (torch.abs(observed_max - expected) < 5e-4)
    assert (torch.abs(observed_min + expected) < 5e-4)
