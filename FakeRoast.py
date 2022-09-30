import torch
import torch.nn as nn
import numpy as np
from math import sqrt
import pdb

class FakeRoast(nn.Module):
    def __init__(self, W_shape, is_global, weight=None, init_scale=None, compression=None):
        super(FakeRoast, self).__init__()
        self.is_global = is_global
        if is_global:
            assert(weight is not None)
            assert(init_scale is not None)
            self.weight = weight
            self.wsize = weight.numel()
            self.init_scale = init_scale
        else:
            assert(compression is not None)
            self.wsize = int(np.prod(W_shape) * compression)
            self.weight = nn.Parameter(torch.zeros(self.wsize, dtype=torch.float), requires_grad=True)
            self.init_scale = 0.0001
            nn.init.uniform_(self.weight.data, a=-self.init_scale, b = self.init_scale)
        self.W_shape = W_shape
        self.IDX = nn.Parameter(torch.randint(0, self.wsize, size=W_shape, dtype=torch.int64), requires_grad=False)
        self.G = nn.Parameter(torch.randint(0, 2, size=W_shape, dtype=torch.float)*2 - 1, requires_grad=False)

    def forward(self):
        W = torch.mul(self.weight[self.IDX], self.G)
        return W

    def grad_comp_to_orig(self, grad): # grad of compressed to original
        return torch.mul(grad[self.IDX],self.G)

    def grad_orig_to_comp(self, grad): # original gradient to compressed gradient . 
        out_grad = torch.zeros(self.wsize, dtype=torch.float, device=grad.device)
        out_grad.scatter_add_(0, self.IDX.view(-1), (torch.mul(grad, self.G)).view(-1))

        count = torch.zeros(self.wsize, dtype=torch.float, device=grad.device)
        count.scatter_add_(0, self.IDX.view(-1), torch.ones_like(self.IDX, device=grad.device, dtype=torch.float).view(-1))
        return (out_grad, count)

    def wt_comp_to_orig(self, wt):
        return torch.mul(wt[self.IDX],self.G)

    def wt_orig_to_comp(self, wt):
        out_wt = torch.zeros(self.wsize, dtype=torch.float, device=wt.device)
        out_wt.scatter_add_(0, self.IDX.view(-1), (torch.mul(wt, self.G)).view(-1))

        count = torch.zeros(self.wsize, dtype=torch.float, device=wt.device)
        count.scatter_add_(0, self.IDX.view(-1), torch.ones_like(self.IDX, device=wt.device, dtype=torch.float).view(-1)) + 1e-3
        return (out_wt, count)



class FakeRoastLinear(nn.Module):
    def __init__(self, input, output, bias, is_global, weight, init_scale, compression):
        super(FakeRoastLinear, self).__init__()
        self.W_shape = (output, input)
        self.idim = input
        self.odim = output

        self.WHelper = FakeRoast(self.W_shape, is_global, weight, init_scale, compression)
        self.scale = (1/sqrt(self.idim)) / self.WHelper.init_scale
        self.bias = None
        if bias :
            self.bias = nn.Parameter(torch.zeros(self.odim, dtype=torch.float), requires_grad = True)

    def forward(self, x):
        W = self.WHelper() * self.scale
        x = nn.functional.linear(x, W, self.bias)
        return x



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
                    padding_mode='zeros'):
        super(FakeRoastConv2d, self).__init__()
        
        
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.is_bias = bias
        W_shape = (out_channels, int(in_channels/groups), kernel_size[0], kernel_size[1])
        self.WHelper = FakeRoast(W_shape, is_global, weight, init_scale, compression)
        
        k = 1.0 * groups / (in_channels * np.prod(kernel_size))
        self.scale = sqrt(k) / self.WHelper.init_scale
        self.bias = None
        if self.is_bias :
            self.bias = nn.paramter(torch.zeros(out_channels))

    def forward(self, x):
        W = self.WHelper() * self.scale
        x = torch.nn.functional.conv2d(x, W, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return x


class FakeRoastEmbedding(nn.Module):
      def __init__(self, num_embeddings, embedding_dim,
                    is_global, weeight=None, init_scale=None, compression=None,
                    padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
          super(FakeRoastEmbedding, self).__init__()
          W_shape = (num_embeddings, embedding_dim)
          self.WHelper = FakeRoast(W_shape, is_global, weeight, init_scale, compression)
          
          self.scale = sqrt(1. / num_embeddings) / self.WHelper.init_scale
          self.padding_idx = padding_idx
          self.max_norm = max_norm
          self.norm_type = norm_type
          self.scale_grad_by_freq = scale_grad_by_freq
          self.sparse = sparse


      def forward(self, x):
          W = self.WHelper() * self.scale
          return nn.functional.embedding(x, W, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
