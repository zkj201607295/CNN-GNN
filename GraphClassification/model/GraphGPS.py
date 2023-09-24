import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GraphConv, TopKPooling

from torch_geometric.nn import GCNConv, JumpingKnowledge,SAGEConv

from gps_conv import GPSConv

from torch_geometric.nn import global_add_pool


class Graphgps(torch.nn.Module):
    def __init__(self, dataset, num_layers, args):
        super().__init__()
        self.conv1 = GPSConv(args.hidden2, GCNConv(args.hidden2, args.hidden2, bias=False), 4)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GPSConv(args.hidden2, GCNConv(args.hidden2, args.hidden2, bias=False), 4))
        self.lin1 = Linear(dataset.num_features, args.hidden2)
        self.lin2 = Linear(args.hidden2, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.lin1(x))
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = self.lin2(x)
        x = global_add_pool(x, batch)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__