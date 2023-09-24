import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from methods import *


def make_convolution(in_channels, out_channels):
    return GINConv(nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Linear(out_channels, out_channels),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    ))

# Define a pooling layer for centrality features
class CentPool(nn.Module):
    def __init__(self, in_dim, ratio, p):
        super(CentPool, self).__init__()
        self.ratio = ratio
        self.cent_num = 6
        self.sigmoid = nn.Sigmoid()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.feature_proj = nn.Linear(in_dim, 1, device=device)
        self.structure_proj = nn.Linear(self.cent_num, 1, device=device)
        self.final_proj = nn.Linear(2, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, edge_index, h):
        #print(edge_index.is_cuda)
        device = edge_index.device
        Z = self.drop(h)
        Z = Z.to(device)
        G = edge_index_to_nx_graph(edge_index, h.shape[0])
        C = all_centralities(G)
        C = C.to(device)
        #print(C.is_cuda)
        feature_weights = self.feature_proj(Z)
        structure_weights = self.structure_proj(C)
        weights = self.final_proj(
            torch.cat([feature_weights, structure_weights], dim=1)).squeeze()  # Combine and project weights
        scores = self.sigmoid(weights)
        g, h, idx = top_k_pool(scores, edge_index, h, self.ratio)
        edge_index = edge_index[:, idx]
        return g, h, idx, edge_index


# Define a pooling layer for spectral features
class SpectPool(nn.Module):
    def __init__(self, in_dim, ratio, p):
        super(SpectPool, self).__init__()
        self.ratio = ratio
        self.eigs_num = 3
        self.sigmoid = nn.Sigmoid()
        self.feature_proj = nn.Linear(in_dim, 1)
        self.structure_proj = nn.Linear(self.eigs_num, 1)
        self.final_proj = nn.Linear(2, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, edge_index, h):
        Z = self.drop(h)
        G = edge_index_to_nx_graph(edge_index, h.shape[0])
        L = normalized_laplacian(G)
        L_a = approximate_matrix(L, self.eigs_num)
        feature_weights = self.feature_proj(Z)
        structure_weights = self.structure_proj(L_a)
        weights = self.final_proj(
            torch.cat([feature_weights, structure_weights], dim=1)).squeeze()  # Combine and project weights
        scores = self.sigmoid(weights)
        g, h, idx = top_k_pool(scores, edge_index, h, self.ratio)
        edge_index = edge_index[:, idx]
        return g, h, idx, edge_index


class SimpleUnpool(nn.Module):
    def forward(self, g, h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return new_h


class Unpool(nn.Module):
    def forward(self, g, h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        idx_prime = torch.tensor([index for index in idx if index not in range(g.shape[0])])

        for i in idx_prime:
            normalized_idx = idx.float() / g[i].sum()  # Normalize indices
            weighted_mean = torch.sum(g[i][i] * normalized_idx)  # Compute weighted mean
            new_h[i] = weighted_mean * h

        return new_h


# Creating model that uses centralities
class GIUNetSpect(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GIUNetSpect, self).__init__()

        self.conv1 = make_convolution(num_features, 32)
        self.pool1 = SpectPool(32, ratio=0.8, p=0.5)  # Custom pooling layer

        self.conv2 = make_convolution(32, 64)
        self.pool2 = SpectPool(64, ratio=0.8, p=0.5)  # Custom pooling layer

        self.midconv = make_convolution(64, 64)

        self.decoder2 = make_convolution(64, 32)
        self.decoder1 = nn.Linear(32, num_classes)  # Final classification layer

        self.unpool2 = Unpool()  # Unpool layer after decoder2
        self.unpool1 = Unpool()  # Unpool layer after decoder1

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Encoder
        x1 = F.relu(self.conv1(x, edge_index))
        g1, x1_pooled, idx1, edge_index1 = self.pool1(edge_index, x1)

        x2 = F.relu(self.conv2(x1_pooled, edge_index1))
        _, x2_pooled, idx2, edge_index2 = self.pool2(edge_index1, x2)

        # Middle Convolution
        x_m = F.relu(self.midconv(x2_pooled, edge_index2))

        # Decoder
        x_d2 = self.unpool2(g1, x_m, idx2)
        x_d2 = F.relu(self.decoder2(x_d2, edge_index2))

        x_d1 = self.unpool1(adjacency_matrix(edge_index), x_d2, idx1)
        x_d1 = F.relu(self.decoder1(x_d1))

        x_global_pool = global_mean_pool(x_d1, batch)

        return x_global_pool


class GIUNetCent(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GIUNetCent, self).__init__()

        self.conv1 = make_convolution(num_features, 32)
        self.pool1 = CentPool(32, ratio=0.8, p=0.5)  # Custom pooling layer

        self.conv2 = make_convolution(32, 64)
        self.pool2 = CentPool(64, ratio=0.8, p=0.5)  # Custom pooling layer

        self.midconv = make_convolution(64, 64)

        self.decoder2 = make_convolution(64, 32)
        self.decoder1 = nn.Linear(32, num_classes)  # Final classification layer

        self.unpool2 = Unpool()  # Unpool layer after decoder2
        self.unpool1 = Unpool()  # Unpool layer after decoder1

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x1 = F.relu(self.conv1(x, edge_index))
        g1, x1_pooled, idx1, edge_index1 = self.pool1(edge_index, x1)

        x2 = F.relu(self.conv2(x1_pooled, edge_index1))
        _, x2_pooled, idx2, edge_index2 = self.pool2(edge_index1, x2)

        x_m = F.relu(self.midconv(x2_pooled, edge_index2))

        x_d2 = self.unpool2(g1, x_m, idx2)
        x_d2 = F.relu(self.decoder2(x_d2, edge_index2))

        x_d1 = self.unpool1(adjacency_matrix(edge_index), x_d2, idx1)
        x_d1 = F.relu(self.decoder1(x_d1))

        x_global_pool = global_mean_pool(x_d1, batch)

        return x_global_pool