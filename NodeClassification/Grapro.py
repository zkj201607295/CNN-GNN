from typing import Optional
from torch_geometric.typing import OptTensor
import math
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian
from scipy.special import comb
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np


class Gra_inc(MessagePassing):
    def __init__(self, K, bias=True, **kwargs):
        super(Gra_inc, self).__init__(aggr='add', **kwargs)

        self.K = K
        self.temp = Parameter(torch.Tensor(self.K + 1))
        #self.temp = [1.427, 3.069, 3.548]
        self.reset_parameters()
        self._cached = None

    def reset_parameters(self):
        self.temp.data.fill_(1)
        self._cached = None

    def forward(self, x, edge_index, edge_weight=None):
        #TEMP = F.relu(self.temp)
        TEMP = self.temp
        if self._cached is None:
            # L=I-D^(-0.5)AD^(-0.5)
            edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
                                               num_nodes=x.size(self.node_dim))
            # 2I-L
            edge_index2, norm2 = add_self_loops(edge_index1, -norm1, fill_value=2., num_nodes=x.size(self.node_dim))
            self._cached = (edge_index2, norm2)
        else:
            edge_index2, norm2 = self._cached[0], self._cached[1]

        tmp = []
        tmp.append(x)
        out = TEMP[0] * tmp[0]
        for i in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        #out = (comb(self.K, 0) / (2 ** self.K)) * TEMP[0] * tmp[self.K]

        for i in range(self.K):
            #x = tmp[self.K - i - 1]
            #x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            #for j in range(i):
            #    x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            #out = out + (comb(self.K, i + 1) / (2 ** self.K)) * TEMP[i + 1] * x
            out = out + TEMP[i + 1] * tmp[i + 1]
            #print(TEMP)
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)
