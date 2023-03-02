import torch
import torch.nn as nn
import numpy as np
from math import sqrt
import pdb
import mmh3

#N = 100000000 # your wsize should be less than this 
N  = 536870912 # your wsize should be less than this 

# TODO(aditya) vectorize ONLY FOR LOCAL
def idx_circ(W_shape, wsize, seed):
    n = np.prod(W_shape)
    m = wsize
    IDX = torch.arange(n)
    for i in range(IDX.shape[0]):
        h1 = mmh3.hash(str(i), seed=seed) % n
        if h1 < m:
            chunk = i // wsize
            offset = i % wsize
            IDX[i] = (mmh3.hash(str(chunk + 1033),
                      seed=seed*3) + offset) % wsize
        else:
            IDX[i] = mmh3.hash(str(i), seed=seed*3) % wsize
    return IDX.reshape(W_shape)


class FakeRoast(nn.Module):
    def __init__(self,
                 W_shape,
                 is_global,
                 weight=None,
                 init_scale=None,
                 compression=None,
                 test=False,
                 matrix_mode="random",
                 seed=2023):
        super(FakeRoast, self).__init__()
        self.is_global = is_global
        self.seed = seed
        if is_global:
            assert (weight is not None)
            assert (init_scale is not None)
            self.weight = weight
            self.wsize = weight.numel()
            self.init_scale = init_scale
        else:
            assert (compression is not None)
            self.wsize = int(np.prod(W_shape) * compression)
            self.weight = nn.Parameter(torch.zeros(
                self.wsize, dtype=torch.float), requires_grad=True)
            if init_scale is not None:
                self.init_scale = init_scale
            else:
                self.init_scale = 1/sqrt(W_shape[1])
            nn.init.uniform_(self.weight.data, a=-
                             self.init_scale, b=self.init_scale)
        self.W_shape = W_shape

        gen = torch.Generator()
        gen.manual_seed(seed)
        if test:
            print("TESTING ...")
            self.IDX = nn.Parameter(torch.arange(
                np.prod(W_shape)).reshape(W_shape), requires_grad=False)
            assert (self.wsize >= np.prod(W_shape))
        else:
            if matrix_mode == "random":
                # making it consistent for power of 2 compression
                assert(N > self.wsize)
                self.IDX = nn.Parameter(torch.randint(0, N, size=W_shape, dtype=torch.int64, generator=gen) % self.wsize, requires_grad=False)
            elif matrix_mode == "circ_random": 
                assert(not is_global) # fix the idx circ for global if needed
                self.IDX = idx_circ(W_shape, N, seed) % self.wsize
            else:
                raise NotImplementedError
        self.G = nn.Parameter(torch.randint(0, 2, size=W_shape, dtype=torch.float, generator=gen)*2 - 1, requires_grad=False)
        #print(self.IDX.view(-1)[0:5])
        #print(self.G.view(-1)[0:5])

    def forward(self):
        W = torch.mul(self.weight[self.IDX], self.G)
        return W

    def forward_partial(self, x):
        W = torch.mul(self.weight[self.IDX[x]], self.G[x])
        return W
        

    def grad_comp_to_orig(self, grad):  # grad of compressed to original
        return torch.mul(grad[self.IDX], self.G)

    # original gradient to compressed gradient .
    def grad_orig_to_comp(self, grad):
        out_grad = torch.zeros(
            self.wsize, dtype=torch.float, device=grad.device)
        out_grad.scatter_add_(0, self.IDX.view(-1),
                              (torch.mul(grad, self.G)).view(-1))

        count = torch.zeros(self.wsize, dtype=torch.float, device=grad.device)
        count.scatter_add_(0, self.IDX.view(-1), torch.ones_like(self.IDX,
                           device=grad.device, dtype=torch.float).view(-1))
        return (out_grad, count)

    def wt_comp_to_orig(self, wt):
        return torch.mul(wt[self.IDX], self.G)

    def wt_orig_to_comp(self, wt):
        out_wt = torch.zeros(self.wsize, dtype=torch.float, device=wt.device)
        out_wt.scatter_add_(0, self.IDX.view(-1),
                            (torch.mul(wt, self.G)).view(-1))

        count = torch.zeros(self.wsize, dtype=torch.float, device=wt.device)
        count.scatter_add_(0, self.IDX.view(-1), torch.ones_like(self.IDX,
                           device=wt.device, dtype=torch.float).view(-1)) + 1e-3
        return (out_wt, count)


class FakeRoastLinear(nn.Module):
    def __init__(self, input, output, bias, is_global, weight, init_scale, compression, test=False, matrix_mode="random", seed=1024):
        super(FakeRoastLinear, self).__init__()
        self.W_shape = (output, input)
        self.idim = input
        self.odim = output
        self.compression = compression
        self.is_global = is_global
        self.test = test
        self.matrix_mode = matrix_mode
        self.seed = seed

        if is_global == False:
            init_scale = 1/sqrt(self.idim)
        self.WHelper = FakeRoast(
            self.W_shape, is_global, weight, init_scale, compression, test, matrix_mode, seed)
        self.scale = (1/sqrt(self.idim)) / self.WHelper.init_scale
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(
                self.odim, dtype=torch.float), requires_grad=True)

    def forward(self, x):
        W = self.WHelper() * self.scale
        x = nn.functional.linear(x, W, self.bias)
        return x

    def __repr__(self):
        if self.test:
            return "FakeRoastLinearTESTLinearIDX(in={}, out={}, global={}, scale={}, compression={}, testLinearIDX={}, matrix_mode={})".format(self.idim, self.odim, self.is_global, self.scale, self.compression, self.test, self.matrix_mode)
        else:
            return "FakeRoastLinear(in={}, out={}, global={}, scale={}, compression={}, testLinearIDX={}, matrix_mode={}, seed={})".format(self.idim, self.odim, self.is_global, self.scale, self.compression, self.test, self.matrix_mode, self.seed)


class FakeRoastConv2d(nn.Module):
    def __init__(self, in_channels,
                    out_channels,
                    kernel_size,
                    is_global,
                    weight=None,
                    init_scale=None,
                    compression=None,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1, 
                    bias=True,
                    padding_mode='zeros',
                    test=False,
                    matrix_mode="random",
                    seed = 2023):
        super(FakeRoastConv2d, self).__init__()

        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.is_bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.is_global = is_global
        self.compression = compression
        self.test = test
        self.matrix_mode = matrix_mode
        self.seed = seed
        

        W_shape = (out_channels, int(in_channels/groups),
                   kernel_size[0], kernel_size[1])

        k = 1.0 * groups / (in_channels * np.prod(kernel_size))
        if is_global == False:
            init_scale = sqrt(k) 
        self.WHelper = FakeRoast(W_shape, is_global, weight, init_scale, compression, test=test, matrix_mode=matrix_mode, seed=seed)
        
        self.scale = sqrt(k) / self.WHelper.init_scale
        self.bias = None
        if self.is_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        W = self.WHelper() * self.scale
        x = torch.nn.functional.conv2d(x, W, bias=self.bias, stride=self.stride,
                                       padding=self.padding, dilation=self.dilation, groups=self.groups)
        return x

    def __repr__(self):
        if self.test:
            return "FakeRoastConv2dTESTLinearIDX(in={}, out={}, global={}, scale={}, compression={}, testLinearIDX={}, matrix_mode={})".format(self.in_channels, self.out_channels, self.is_global, self.scale, self.compression, self.test, self.matrix_mode)
        else:
            return "FakeRoastConv2d(in={}, out={}, global={}, scale={}, compression={}, testLinearIDX={}, matrix_mode={}, seed={})".format(self.in_channels, self.out_channels, self.is_global, self.scale, self.compression, self.test, self.matrix_mode, self.seed)

class FakeRoastEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 is_global,
                 weight=None,
                 init_scale=None,
                 compression=None,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2.0,
                 scale_grad_by_freq=False,
                 sparse=False):
        super(FakeRoastEmbedding, self).__init__()
        W_shape = (num_embeddings, embedding_dim)
        if is_global == False:
            init_scale = sqrt(1. / num_embeddings)
        self.WHelper = FakeRoast(
            W_shape, is_global, weight, init_scale, compression)
        torch.nn.init.normal_(self.WHelper.weight)
        self.compression = compression
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.scale = sqrt(1. / num_embeddings) / self.WHelper.init_scale
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

    def forward(self, x):
        #W = self.WHelper() * self.scale
        #return nn.functional.embedding(x, W, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        W = self.WHelper.forward_partial(x) * self.scale
        return W
    def __repr__(self):
            return "FakeRoastEmbedding(num_embeddings={}, embedding_dim={}, compression={}, seed={})".format(self.num_embeddings, self.embedding_dim, self.compression, self.WHelper.seed )

class FakeRoastLSTM(nn.Module):
    def __init__(self,
                 input_dim,
                 n_hidden,
                 is_global,
                 init_scale,
                 compression,
                 matrix_mode,
                 weight=None):
        super(FakeRoastLSTM, self).__init__()
        self.input_dim = input_dim
        self.n_hidden = n_hidden

        # WHelper1 is too small for Roast, replaced with W below
        # self.W_shape1 = (input_dim, n_hidden * 4)
        self.W_shape2 = (n_hidden, n_hidden * 4)

        self.W = nn.Parameter(torch.zeros([input_dim, n_hidden * 4], dtype=torch.float32), requires_grad=True)
        # self.U = nn.Parameter(torch.tensor(n_hidden, n_hidden * 4))

        # self.WHelper1 = FakeRoast(
        #     W_shape=self.W_shape1, is_global=is_global, init_scale=init_scale, weight=weight, compression=compression, matrix_mode=matrix_mode)
        self.WHelper2 = FakeRoast(
            W_shape=self.W_shape2, is_global=is_global, init_scale=init_scale, weight=weight, compression=compression, matrix_mode=matrix_mode)

        self.scale = 1. / sqrt(self.n_hidden) / self.WHelper2.init_scale

        self.bias = nn.Parameter(torch.zeros(
            [n_hidden * 4], dtype=torch.float32), requires_grad=True)

        # self.init_scale = 1. / sqrt(self.n_hidden)
        # for weight in self.parameters():
        #     weight.data.uniform_(-self.init_scale, self.init_scale)


    # Forward pass of the LSTM cell.
    #
    # x: the data with shape (batch_size, sequence_size, input_dim)
    def forward(self, x):

        batch_size, sequence_size, _ = x.size()

        hidden_seq = []

        h_t = torch.zeros(batch_size, self.n_hidden,
                          dtype=torch.float32, device="cuda")  # hidden state
        c_t = torch.zeros(batch_size, self.n_hidden,
                          dtype=torch.float32, device="cuda")  # cell state

        for t in range(sequence_size):
            x_t = x[:, t, :]  # get batched values at current time step

            # gates = x_t @ self.W + h_t @ self.U + self.bias

            # gates = x_t @ self.WHelper1() + h_t @ self.WHelper2() + self.bias

            gates = x_t @ self.W + h_t @ self.WHelper2() + self.bias

            i_t, f_t, c_t_pl, o_t = (
                torch.sigmoid(gates[:, :self.n_hidden]),                       # input
                torch.sigmoid(gates[:, self.n_hidden: self.n_hidden * 2]),     # forget
                torch.tanh(gates[:, self.n_hidden * 2: self.n_hidden * 3]),
                torch.sigmoid(gates[:, self.n_hidden * 3:])                    # output
            )

            c_t = f_t * c_t + i_t * c_t_pl
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


class LowRankLinear(nn.Module):
    def __init__(self, input, output, compression, bias):
        super(LowRankLinear, self).__init__()
        self.idim = input
        self.odim = output
        self.compression = compression
        self.intermediate_dim = int(((input * output) * compression) / (input + output))

        assert(self.intermediate_dim > 0)
        self.w1 = nn.Parameter(torch.zeros((self.idim, self.intermediate_dim), dtype=torch.float), requires_grad=True)
        self.w2 = nn.Parameter(torch.zeros((self.intermediate_dim, self.odim), dtype=torch.float), requires_grad=True)
        nn.init.normal_(self.w1.data)
        nn.init.normal_(self.w2.data)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(
                self.odim, dtype=torch.float), requires_grad=True)

    def forward(self, x):
        W = torch.matmul(self.w1, self.w2)
        x = torch.matmul(x, W)
        if self.bias is not None:
            x = x + self.bias
        return x

    def __repr__(self):
        return "LowRankLinear(in={}, int={}, out={})".format(self.idim, self.intermediate_dim, self.odim)

    def wt_orig_to_comp(self, W): 
        # orig is (out,in)
        input = W.T.shape[0]
        output = W.T.shape[1]
        U,S,Vh = torch.svd_lowrank(W.T, q=self.intermediate_dim, niter=10)
        B = torch.matmul(U, torch.sqrt(torch.diag(S)))
        C = torch.matmul(Vh, torch.sqrt(torch.diag(S)))
        return B, C.T

    def wt_comp_to_orig(self, B, C):
        # orig is out,in
        return torch.matmul(B,C).T 



class LowRankEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 compression=None,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2.0,
                 scale_grad_by_freq=False,
                 sparse=False):
        super(LowRankEmbedding, self).__init__()
        self.intermediate_dim= int(((embedding_dim * num_embeddings) * compression) / (num_embeddings + embedding_dim))
        assert(self.intermediate_dim > 0)
        self.w1 = nn.Parameter(torch.zeros(num_embeddings, self.intermediate_dim), requires_grad=True)
        self.w2 = nn.Parameter(torch.zeros(self.intermediate_dim, embedding_dim), requires_grad=True)

        nn.init.normal_(self.w1.data)
        nn.init.xavier_normal_(self.w2.data)
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, x):
        emb = nn.functional.embedding(x, self.w1, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        emb =torch.matmul(emb, self.w2)
        return emb


    def wt_orig_to_comp(self, W):
        U,S,Vh = torch.svd_lowrank(W, q=self.intermediate_dim, niter=10)
        B = torch.matmul(U, torch.sqrt(torch.diag(S)))
        C = torch.matmul(Vh, torch.sqrt(torch.diag(S)))
        return B, C.T

    def wt_comp_to_orig(self, B, C):
        # orig is out,in
        return torch.matmul(B,C)
    
    def __repr__(self):
        return "LowRankEmbedding(in={}, int={}, out={})".format(self.num_embeddings, self.intermediate_dim, self.embedding_dim)
