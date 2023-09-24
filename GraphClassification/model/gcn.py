import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GraphConv, TopKPooling

from torch_geometric.nn import GCNConv, JumpingKnowledge,SAGEConv

from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from model.GraphInception import Gra_inc

class AFGCN(torch.nn.Module):
    def __init__(self, dataset, num_layers, args):
        super(AFGCN, self).__init__()

        self.lin11 = Linear(dataset.num_features, args.hidden1)
        self.lin12 = Linear(dataset.num_features, args.hidden1)
        self.lin13 = Linear(dataset.num_features, args.hidden1)

        self.lin21 = Linear(args.hidden1, args.hidden1)
        self.lin22 = Linear(args.hidden1, args.hidden1)
        self.lin23 = Linear(args.hidden1, args.hidden1)

        self.lin31 = Linear(args.hidden1, args.hidden1)
        self.lin32 = Linear(args.hidden1, args.hidden1)
        self.lin33 = Linear(args.hidden1, args.hidden1)

        self.lin1 = Linear(2 * args.hidden1, args.hidden1)
        self.lin2 = Linear(args.hidden1, args.hidden2)
        self.lin3 = Linear(args.hidden2, dataset.num_classes)

        self.m = torch.nn.BatchNorm1d(dataset.num_classes)

        self.prop11 = Gra_inc(2)
        self.prop12 = Gra_inc(2)
        self.prop13 = Gra_inc(2)

        self.prop21 = Gra_inc(2)
        self.prop22 = Gra_inc(2)
        self.prop23 = Gra_inc(2)

        self.prop31 = Gra_inc(2)
        self.prop32 = Gra_inc(2)
        self.prop33 = Gra_inc(2)

        self.dprate = args.dprate
        self.dropout = args.dropout

        self.pool1 = TopKPooling(args.hidden1, ratio=0.8)
        self.pool2 = TopKPooling(args.hidden1, ratio=0.8)
        self.pool3 = TopKPooling(args.hidden1, ratio=0.8)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.lin11.reset_parameters()
        self.lin12.reset_parameters()
        self.lin13.reset_parameters()
        self.lin21.reset_parameters()
        self.lin22.reset_parameters()
        self.lin23.reset_parameters()
        self.lin31.reset_parameters()
        self.lin32.reset_parameters()
        self.lin33.reset_parameters()
        self.prop11.reset_parameters()
        self.prop12.reset_parameters()
        self.prop13.reset_parameters()
        self.prop21.reset_parameters()
        self.prop22.reset_parameters()
        self.prop23.reset_parameters()
        self.prop31.reset_parameters()
        self.prop32.reset_parameters()
        self.prop33.reset_parameters()


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x11 = F.relu(self.lin11(x))
        x12 = F.relu(self.lin12(x))
        x13 = F.relu(self.lin13(x))

        x11 = self.prop13(x11, edge_index)
        x12 = self.prop11(x12, edge_index)
        x13 = self.prop12(x13, edge_index)
        x = x11 + x12 + x13

        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x21 = F.relu(self.lin21(x))
        x22 = F.relu(self.lin22(x))
        x23 = F.relu(self.lin23(x))

        x21 = self.prop23(x21, edge_index)
        x22 = self.prop21(x22, edge_index)
        x23 = self.prop22(x23, edge_index)
        x = x21 + x22 + x23

        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x31 = F.relu(self.lin31(x))
        x32 = F.relu(self.lin32(x))
        x33 = F.relu(self.lin33(x))

        x31 = self.prop33(x31, edge_index)
        x32 = self.prop31(x32, edge_index)
        x33 = self.prop32(x33, edge_index)
        x = x31 + x32 + x33

        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

class ConvG(torch.nn.Module):
    def __init__(self, dataset, num_layers, args):
        super(ConvG, self).__init__()

        #self.lin11 = Linear(dataset.num_features, args.hidden1)
        self.lin12 = Linear(dataset.num_features, args.hidden1)
        #self.lin13 = Linear(dataset.num_features, args.hidden1)

        #self.lin21 = Linear(args.hidden1, args.hidden1)
        self.lin22 = Linear(args.hidden1, args.hidden1)
        #self.lin23 = Linear(args.hidden1, args.hidden1)

        #self.lin31 = Linear(args.hidden1, args.hidden1)
        self.lin32 = Linear(args.hidden1, args.hidden1)
        #self.lin33 = Linear(args.hidden1, args.hidden1)

        self.lin1 = Linear(2 * args.hidden1, args.hidden1)
        self.lin2 = Linear(args.hidden1, args.hidden2)
        self.lin3 = Linear(args.hidden2, dataset.num_classes)

        self.m = torch.nn.BatchNorm1d(dataset.num_classes)

        self.prop11 = Gra_inc(2)
        #self.prop12 = Gra_inc(2)

        self.prop21 = Gra_inc(2)
        #self.prop22 = Gra_inc(2)

        self.prop31 = Gra_inc(2)
        #self.prop32 = Gra_inc(2)

        self.dprate = args.dprate
        self.dropout = args.dropout

        self.pool1 = TopKPooling(args.hidden1, ratio=0.8)
        self.pool2 = TopKPooling(args.hidden1, ratio=0.8)
        self.pool3 = TopKPooling(args.hidden1, ratio=0.8)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        #self.lin11.reset_parameters()
        self.lin12.reset_parameters()
        #self.lin13.reset_parameters()
        #self.lin21.reset_parameters()
        self.lin22.reset_parameters()
        #self.lin23.reset_parameters()
        #self.lin31.reset_parameters()
        self.lin32.reset_parameters()
        #self.lin33.reset_parameters()
        self.prop11.reset_parameters()
        #self.prop12.reset_parameters()
        self.prop21.reset_parameters()
        #self.prop22.reset_parameters()
        self.prop31.reset_parameters()
        #self.prop32.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        #x11 = F.relu(self.lin11(x))
        x12 = F.relu(self.lin12(x))
        #x13 = F.relu(self.lin13(x))

        x12 = self.prop11(x12, edge_index)
        #x13 = self.prop12(x13, edge_index)
        #x = x11 + x12 + x13
        x = x12

        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        #x21 = F.relu(self.lin21(x))
        x22 = F.relu(self.lin22(x))
        #x23 = F.relu(self.lin23(x))

        x22 = self.prop21(x22, edge_index)
        #x23 = self.prop22(x23, edge_index)
        #x = x21 + x22 + x23
        x = x22

        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        #x31 = F.relu(self.lin31(x))
        x32 = F.relu(self.lin32(x))
        #x33 = F.relu(self.lin33(x))

        x32 = self.prop31(x32, edge_index)
        #x33 = self.prop32(x33, edge_index)
       # x = x31 + x32 + x33
        x = x32

        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

class GCN(torch.nn.Module):
    def __init__(self, dataset, num_layers, args):
        super(GCN, self).__init__()

        self.conv1 = GraphConv(dataset.num_features, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)

        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

class GCNWithJK(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, mode='cat'):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]
        x = self.jump(xs)
        x = gmp(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
