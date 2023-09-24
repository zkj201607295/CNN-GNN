from typing import Optional

from torch_geometric.nn.inits import glorot, zeros
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


class AF_GCN(MessagePassing):
    def __init__(self, K: int, in_channels: int, out_channels: int,
                 bias: bool = True, **kwargs):
        super(AF_GCN, self).__init__(aggr='add', **kwargs)

        self.K = K

        self.t = Parameter(torch.Tensor(self.K))

        self.temp = torch.nn.ParameterList()
        for i in range (K):
            self.temp.append(Parameter(torch.Tensor(3)))

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = torch.nn.ParameterList()
        for i in range (K):
            self.weight.append(Parameter(torch.Tensor(in_channels, out_channels)))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for i in range (self.K):
            glorot(self.weight[i])
            self.temp[i].data.fill_(1)

        self.t.data.fill_(1)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        #TEMP = F.relu(self.temp)
        TEMP = self.temp

        # L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
                                           num_nodes=x.size(self.node_dim))
        # 2I-L
        edge_index2, norm2 = add_self_loops(edge_index1, -norm1, fill_value=2., num_nodes=x.size(self.node_dim))

        sum_temp = []
        sum_t = self.t[0] + self.t[1] + self.t[2]
        for i in range(self.K):
            sum_temp.append(TEMP[i][0] + TEMP[i][1] + TEMP[i][2])

        tmp = []
        for i in range(self.K):
            tmp.append(torch.matmul(x, self.weight[i]))

        out_m = []
        for i in range(self.K):
            x = tmp[i]
            xxx = (TEMP[i][0] / sum_temp[i]) * x
            for j in range(1, 3):
                x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
                xxx = xxx + (TEMP[i][j] / sum_temp[i]) * x
            out_m.append(xxx)

        out = (self.t[0]/sum_t) * out_m[0]
        for i in range(self.K - 1):
            out = out + (self.t[i + 1]/sum_t) * out_m[i + 1]

        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)
