import torch
import math
import pickle
import os.path as osp
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import Amazon, AttributedGraphDataset, WikipediaNetwork
from torch_geometric.datasets import AmazonProducts,Coauthor,Reddit,Yelp,Reddit2, Flickr
from torch_geometric.nn import APPNP
from torch_geometric.utils import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import is_undirected, to_undirected
from torch_geometric.io import read_npz

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


class WebKB(InMemoryDataset):

    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
           'master/new_data')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)

from typing import Callable, List, Optional


class Actor(InMemoryDataset):

    url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt'
                ] + [f'film_split_0.6_0.2_{i}.npz' for i in range(10)]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        for f in self.raw_file_names[:2]:
            download_url(f'{self.url}/new_data/film/{f}', self.raw_dir)
        for f in self.raw_file_names[2:]:
            download_url(f'{self.url}/splits/{f}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = [x.split('\t') for x in f.read().split('\n')[1:-1]]

            rows, cols = [], []
            for n_id, col, _ in data:
                col = [int(x) for x in col.split(',')]
                rows += [int(n_id)] * len(col)
                cols += col
            row, col = torch.tensor(rows), torch.tensor(cols)

            x = torch.zeros(int(row.max()) + 1, int(col.max()) + 1)
            x[row, col] = 1.
            print(len(data))
            y = torch.empty(len(data), dtype=torch.long)
            for n_id, _, label in data:
                y[int(n_id)] = int(label)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            edge_index = coalesce(edge_index, num_nodes=x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

class WikipediaNetwork(InMemoryDataset):

    raw_url = 'https://graphmining.ai/datasets/ptg/wiki'
    processed_url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/'
                     'geom-gcn/f1fc0d14b3b019c562737240d06ec83b07d16a8f')

    def __init__(self, root: str, name: str, geom_gcn_preprocess: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        self.geom_gcn_preprocess = geom_gcn_preprocess
        assert self.name in ['chameleon', 'crocodile', 'squirrel']
        if geom_gcn_preprocess and self.name == 'crocodile':
            raise AttributeError("The dataset 'crocodile' is not available in "
                                 "case 'geom_gcn_preprocess=True'")
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        if self.geom_gcn_preprocess:
            return osp.join(self.root, self.name, 'geom_gcn', 'raw')
        else:
            return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        if self.geom_gcn_preprocess:
            return osp.join(self.root, self.name, 'geom_gcn', 'processed')
        else:
            return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        if self.geom_gcn_preprocess:
            return (['out1_node_feature_label.txt', 'out1_graph_edges.txt'] +
                    [f'{self.name}_split_0.6_0.2_{i}.npz' for i in range(10)])
        else:
            return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        if self.geom_gcn_preprocess:
            for filename in self.raw_file_names[:2]:
                url = f'{self.processed_url}/new_data/{self.name}/{filename}'
                download_url(url, self.raw_dir)
            for filename in self.raw_file_names[2:]:
                url = f'{self.processed_url}/splits/{filename}'
                download_url(url, self.raw_dir)
        else:
            download_url(f'{self.raw_url}/{self.name}.npz', self.raw_dir)

    def process(self):
        if self.geom_gcn_preprocess:
            with open(self.raw_paths[0], 'r') as f:
                data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)
            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

            with open(self.raw_paths[1], 'r') as f:
                data = f.read().split('\n')[1:-1]
                data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            edge_index = coalesce(edge_index, num_nodes=x.size(0))

            train_masks, val_masks, test_masks = [], [], []
            for filepath in self.raw_paths[2:]:
                f = np.load(filepath)
                train_masks += [torch.from_numpy(f['train_mask'])]
                val_masks += [torch.from_numpy(f['val_mask'])]
                test_masks += [torch.from_numpy(f['test_mask'])]
            train_mask = torch.stack(train_masks, dim=1).to(torch.bool)
            val_mask = torch.stack(val_masks, dim=1).to(torch.bool)
            test_mask = torch.stack(test_masks, dim=1).to(torch.bool)

            data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                        val_mask=val_mask, test_mask=test_mask)

        else:
            data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
            x = torch.from_numpy(data['features']).to(torch.float)
            edge_index = torch.from_numpy(data['edges']).to(torch.long)
            edge_index = edge_index.t().contiguous()
            edge_index = to_undirected(edge_index)
            edge_index = coalesce(edge_index, num_nodes=x.size(0))
            y = torch.from_numpy(data['target']).to(torch.float)

            data = Data(x=x, edge_index=edge_index, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])



def DataLoader(name):
    
    name = name.lower()
    if name in ['cora', 'citeseer', 'pubmed']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures())
    elif name in ['computers', 'photo']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Amazon(path, name, T.NormalizeFeatures())

    elif name in ['actor']:
        dataset = Actor(root='./data/film',transform=T.NormalizeFeatures())

    elif name in ['texas', 'cornell', 'wisconsin']:
        dataset = WebKB(root='./data/',name=name, transform=T.NormalizeFeatures())

    elif name in ['cs', 'physics']:
        dataset = Coauthor(root='./data/',name=name, transform=T.NormalizeFeatures())

    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root='./data/',name=name)

    elif name in ['reddit']:
        dataset = Reddit(root='./data/Reddit', transform=T.ToUndirected())

    elif name in ['reddit2']:
        dataset = Reddit2(root='./data/Reddit2')

    elif name in ['flickr']:
        dataset = Flickr(root='./data/Flickr')

    elif name in ['ogbn-arxiv']:
        dataset = PygNodePropPredDataset(root='./data/', name=name, transform=T.ToUndirected())

    elif name in ['ogbn-products']:
        dataset = PygNodePropPredDataset(root='./data/', name=name, transform=T.ToUndirected())

    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    return dataset
