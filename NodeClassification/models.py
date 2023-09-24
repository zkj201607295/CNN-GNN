import torch
import random
import math
import torch.nn.functional as F
import torch_geometric.nn as G
import os.path as osp
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv, SAGEConv, AntiSymmetricConv
from torch_geometric.nn import MessagePassing, APPNP, SGConv, FAConv, DeepGCNLayer
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
from scipy.special import comb
from DirGNNConv import  DirGNNConv
from Bernpro import Bern_prop
from Grapro import  Gra_inc
from AFGCNpro import AF_GCN

class ChebNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(dataset.num_features, 32, K=2)
        self.conv2 = ChebConv(32, dataset.num_classes, K=2)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class ResGCN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(ResGCN, self).__init__()
        self.conv = DeepGCNLayer(dropout = args.dropout)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class AFGCN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(AFGCN, self).__init__()
        self.conv11 = Gra_inc(args.Inc2)
        self.conv12 = Gra_inc(args.Inc2)
        self.conv13 = Gra_inc(args.Inc2)
        self.conv21 = Gra_inc(args.Inc2)
        self.conv22 = Gra_inc(args.Inc2)
        self.conv23 = Gra_inc(args.Inc2)
        self.lin11 = Linear(dataset.num_features, args.hidden)
        self.lin12 = Linear(dataset.num_features, args.hidden)
        self.lin13 = Linear(dataset.num_features, args.hidden)
        #self.lin1 = Linear(args.hidden, dataset.num_classes)
        self.lin21 = Linear(args.hidden, dataset.num_classes)
        self.lin22 = Linear(args.hidden, dataset.num_classes)
        self.lin23 = Linear(args.hidden, dataset.num_classes)
        self.dropout = args.dropout
        self.dprate = args.dprate

    def reset_parameters(self):
        self.conv11.reset_parameters()
        self.conv21.reset_parameters()
        self.conv12.reset_parameters()
        self.conv22.reset_parameters()
        self.conv13.reset_parameters()
        self.conv23.reset_parameters()


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        temp1 = Parameter(torch.Tensor(3))
        temp2 = Parameter(torch.Tensor(3))
        temp1.data.fill_(1)
        temp2.data.fill_(1)
        x11 = F.dropout(x, p=self.dropout, training=self.training)
        x11 = F.relu(self.lin11(x11))
        #x11 = F.dropout(x11, p=self.dropout, training=self.training)
        x12 = F.dropout(x, p=self.dropout, training=self.training)
        x12 = F.relu(self.lin12(x12))
        #x12 = F.dropout(x12, p=self.dropout, training=self.training)
        x13 = F.dropout(x, p=self.dropout, training=self.training)
        x13 = F.relu(self.lin13(x13))
        #x13 = F.dropout(x13, p=self.dropout, training=self.training)
        x11 = self.conv11(x11, edge_index)
        x12 = self.conv12(x12, edge_index)
        x13 = self.conv13(x13, edge_index)
        temp1_sum = temp1[0] + temp1[1] + temp1[2]
        x = (temp1[0]/temp1_sum)*x11 + (temp1[1]/temp1_sum)*x12 + (temp1[2]/temp1_sum)*x13
        # x = self.lin1(x)
        x21 = F.dropout(x, p=self.dprate, training=self.training)
        x21 = F.relu(self.lin21(x21))
        x22 = F.dropout(x, p=self.dprate, training=self.training)
        x22 = F.relu(self.lin22(x22))
        x23 = F.dropout(x, p=self.dprate, training=self.training)
        x23 = F.relu(self.lin23(x23))
        x21 = self.conv21(x21, edge_index)
        x22 = self.conv22(x22, edge_index)
        x23 = self.conv23(x23, edge_index)
        temp2_sum = temp2[0] + temp2[1] + temp2[2]
        x = (temp2[0]/temp2_sum)*x21 + (temp2[1]/temp2_sum)*x22 + (temp2[2]/temp2_sum)*x23

        return F.log_softmax(x, dim=1)

class AFGCN1(torch.nn.Module):
    def __init__(self, dataset, args):
        super(AFGCN1, self).__init__()
        self.conv11 = ChebConv(dataset.num_features, 32, K=2)
        self.conv12 = ChebConv(dataset.num_features, 32, K=2)
        self.conv13 = ChebConv(dataset.num_features, 32, K=2)
        self.conv21 = ChebConv(32, dataset.num_classes, K=2)
        self.conv22 = ChebConv(32, dataset.num_classes, K=2)
        self.conv23 = ChebConv(32, dataset.num_classes, K=2)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv11.reset_parameters()
        self.conv21.reset_parameters()
        self.conv12.reset_parameters()
        self.conv22.reset_parameters()
        self.conv13.reset_parameters()
        self.conv23.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x11 = self.conv11(x, edge_index)
        x12 = self.conv12(x, edge_index)
        x13 = self.conv13(x, edge_index)
        x = F.relu(x11 + x12 + x13)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x21 = self.conv21(x, edge_index)
        x22 = self.conv22(x, edge_index)
        x23 = self.conv23(x, edge_index)
        x = x21 + x22 + x23
        return F.log_softmax(x, dim=1)
'''
class AFGCN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(AFGCN, self).__init__()

        self.conv11 = GCNConv(dataset.num_features, 32)
        self.conv12 = GCNConv(dataset.num_features, 32)
        self.conv13 = GCNConv(dataset.num_features, 32)
        self.conv21 = GCNConv(32, dataset.num_classes)
        self.conv22 = GCNConv(32, dataset.num_classes)
        self.conv23 = GCNConv(32, dataset.num_classes)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv11.reset_parameters()
        self.conv21.reset_parameters()
        self.conv12.reset_parameters()
        self.conv22.reset_parameters()
        self.conv13.reset_parameters()
        self.conv23.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x11 = self.conv11(x, edge_index)
        x12 = self.conv12(x, edge_index)
        x13 = self.conv13(x, edge_index)
        x = F.relu(x11 + x12 + x13)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x21 = self.conv21(x, edge_index)
        x22 = self.conv22(x, edge_index)
        x23 = self.conv23(x, edge_index)
        x = x21 + x22 + x23
        return F.log_softmax(x, dim=1)
'''


class FA_NET(torch.nn.Module):
    def __init__(self, dataset, args):
        super(FA_NET, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.conv1 = FAConv(args.hidden, eps=0.3)
        self.conv2 = FAConv(args.hidden, eps=0.3)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.lin1(x))
        x1 = self.conv1(x, x, edge_index)
        x2 = self.conv2(x1, x, edge_index)
        #x = F.dropout(x2, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[-1] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)
class GPRGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)

        self.prop1 = GPR_prop(args.K, args.alpha, args.Init)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)

class SIGN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(SIGN, self).__init__()
        self.conv1 = SGConv(dataset.num_features, args.hidden, K=0)
        self.conv2 = SGConv(dataset.num_features, args.hidden, K=1)
        self.conv3 = SGConv(dataset.num_features, args.hidden, K=2)
        self.lin = Linear(96, dataset.num_classes)
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x, edge_index))
        x3 = F.relu(self.conv3(x, edge_index))

        x = torch.cat([x1,x2,x3], dim=1)

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin(x)
        return F.log_softmax(x, dim=1)


class GraInc(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GraInc, self).__init__()
        self.lin11 = Linear(dataset.num_features, args.hidden)
        self.lin12 = Linear(dataset.num_features, args.hidden)
        self.lin13 = Linear(dataset.num_features, args.hidden)
        self.lin21 = Linear(dataset.num_features, args.hidden)
        self.lin22 = Linear(dataset.num_features, args.hidden)
        self.lin23 = Linear(dataset.num_features, args.hidden)
        self.lin31 = Linear(dataset.num_features, args.hidden)
        self.lin32 = Linear(dataset.num_features, args.hidden)
        self.lin33 = Linear(dataset.num_features, args.hidden)

        self.lin1 = Linear(3 * args.hidden, dataset.num_classes)

        self.m = torch.nn.BatchNorm1d(args.hidden)

        self.prop21 = Gra_inc(args.Inc1)
        self.prop22 = Gra_inc(args.Inc1)
        self.prop23 = Gra_inc(args.Inc1)
        self.prop31 = Gra_inc(args.Inc3)
        self.prop32 = Gra_inc(args.Inc3)
        self.prop33 = Gra_inc(args.Inc3)


        self.dprate = args.dprate
        self.dropout = args.dropout


    def reset_parameters(self):
        self.prop21.reset_parameters()
        self.prop22.reset_parameters()
        self.prop23.reset_parameters()
        self.prop31.reset_parameters()
        self.prop32.reset_parameters()
        self.prop33.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x11 = F.dropout(x, p=self.dropout, training=self.training)
        x11 = F.relu(self.lin11(x11))
        x12 = F.dropout(x, p=self.dropout, training=self.training)
        x12 = F.relu(self.lin12(x12))
        x13 = F.dropout(x, p=self.dropout, training=self.training)
        x13 = F.relu(self.lin13(x13))

        x21 = F.dropout(x, p=self.dropout, training=self.training)
        x21 = F.relu(self.lin21(x21))
        x22 = F.dropout(x, p=self.dropout, training=self.training)
        x22 = F.relu(self.lin22(x22))
        x23 = F.dropout(x, p=self.dropout, training=self.training)
        x23 = F.relu(self.lin23(x23))

        x31 = F.dropout(x, p=self.dropout, training=self.training)
        x31 = F.relu(self.lin31(x31))
        x32 = F.dropout(x, p=self.dropout, training=self.training)
        x32 = F.relu(self.lin32(x32))
        x33 = F.dropout(x, p=self.dropout, training=self.training)
        x33 = F.relu(self.lin33(x33))

        if self.dprate == 0.0:
            x21 = self.prop21(x21, edge_index)
            x22 = self.prop22(x22, edge_index)
            x23 = self.prop23(x23, edge_index)
            x31 = self.prop31(x31, edge_index)
            x32 = self.prop32(x32, edge_index)
            x33 = self.prop33(x33, edge_index)
            x1 = torch.cat([x11,x12,x13],dim=1)
            x2 = torch.cat([x21,x22,x23],dim=1)
            x3 = torch.cat([x31, x32, x33], dim=1)
            x = x1 + x2 + x3
            x = self.lin1(x)
            return F.log_softmax(x, dim=1)
        else:
            x11 = F.dropout(x11, p=self.dprate, training=self.training)
            x12 = F.dropout(x12, p=self.dprate, training=self.training)
            x13 = F.dropout(x13, p=self.dprate, training=self.training)
            x21 = F.dropout(x21, p=self.dprate, training=self.training)
            x22 = F.dropout(x22, p=self.dprate, training=self.training)
            x23 = F.dropout(x23, p=self.dprate, training=self.training)
            x31 = F.dropout(x31, p=self.dprate, training=self.training)
            x32 = F.dropout(x32, p=self.dprate, training=self.training)
            x33 = F.dropout(x33, p=self.dprate, training=self.training)

            x21 = self.prop21(x21, edge_index)
            x22 = self.prop22(x22, edge_index)
            x23 = self.prop23(x23, edge_index)
            x31 = self.prop31(x31, edge_index)
            x32 = self.prop32(x32, edge_index)
            x33 = self.prop33(x33, edge_index)
            x1 = torch.cat([x11, x12, x13], dim=1)
            x2 = torch.cat([x21, x22, x23], dim=1)
            x3 = torch.cat([x31, x32, x33], dim=1)
            x = x1 + x2 + x3
            x = self.lin1(x)
            return F.log_softmax(x, dim=1)

class ConvG(torch.nn.Module):
    def __init__(self, dataset, args):
        super(ConvG, self).__init__()
        self.lin11 = Linear(dataset.num_features, args.hidden)
        #self.lin12 = Linear(dataset.num_features, args.hidden)
        #self.lin13 = Linear(dataset.num_features, args.hidden)

        #self.lin1 = Linear(3 * args.hidden, dataset.num_classes)
        self.lin1 = Linear(args.hidden, dataset.num_classes)

        self.m = torch.nn.BatchNorm1d(args.hidden)

        self.prop11 = Gra_inc(args.Inc2)
        #self.prop12 = Gra_inc(args.Inc2)
        #self.prop13 = Gra_inc(args.Inc2)

        self.dprate = args.dprate
        self.dropout = args.dropout


    def reset_parameters(self):
        self.prop11.reset_parameters()
        #self.prop12.reset_parameters()
        #self.prop13.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x11 = F.dropout(x, p=self.dropout, training=self.training)
        x11 = F.relu(self.lin11(x11))
        #x12 = F.dropout(x, p=self.dropout, training=self.training)
        #x12 = F.relu(self.lin12(x12))
        #x13 = F.dropout(x, p=self.dropout, training=self.training)
        #x13 = F.relu(self.lin13(x13))

        if self.dprate == 0.0:
            x11 = self.prop11(x11, edge_index)
            #x12 = self.prop12(x12, edge_index)
            #x13 = self.prop13(x13, edge_index)
            #x = torch.cat([x11,x12,x13],dim=1)
            x = self.lin1(x11)
            return F.log_softmax(x, dim=1)
        else:
            x11 = F.dropout(x11, p=self.dprate, training=self.training)
            #x12 = F.dropout(x12, p=self.dprate, training=self.training)
            #x13 = F.dropout(x13, p=self.dprate, training=self.training)

            x11 = self.prop11(x11, edge_index)
            #x12 = self.prop12(x12, edge_index)
            #x13 = self.prop13(x13, edge_index)
            #x = torch.cat([x11, x12, x13], dim=1)
            x = self.lin1(x11)
            return F.log_softmax(x, dim=1)

class BernNet(torch.nn.Module):
    def __init__(self,dataset, args):
        super(BernNet, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.m = torch.nn.BatchNorm1d(dataset.num_classes)
        self.prop1 = Bern_prop(args.K)

        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        #x= self.m(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)

class GCN_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class AntiSymmetric_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(AntiSymmetric_Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.conv1 = AntiSymmetricConv(args.hidden)
        self.conv2 = AntiSymmetricConv(dataset.num_classes)
        self.dropout = args.dropout

    def reset_parameters(self):
        #self.conv1.reset_parameters()
        #self.conv2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class DirGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(DirGNN, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.conv1 = DirGNNConv(GCNConv(args.hidden, args.hidden, bias=False))
        self.conv2 = DirGNNConv(GCNConv(dataset.num_classes, dataset.num_classes, bias=False))
        self.dropout = args.dropout

    def reset_parameters(self):
        #self.conv1.reset_parameters()
        #self.conv2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class SAGE_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(SAGE_Net, self).__init__()
        self.conv1 = SAGEConv(dataset.num_features, args.hidden)
        self.conv2 = SAGEConv(args.hidden, dataset.num_classes)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class SGC_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(SGC_Net, self).__init__()
        self.conv1 = SGConv(dataset.num_features, dataset.num_classes, K=2)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GAT_Net, self).__init__()
        self.conv1 = GATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)
        self.conv2 = GATConv(
            args.hidden * args.heads,
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class APPNP_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


class MLP(torch.nn.Module):
    def __init__(self, dataset,args):
        super(MLP, self).__init__()

        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.dropout =args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)
