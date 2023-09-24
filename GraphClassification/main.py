import argparse
from itertools import product

from datasets import get_dataset
from model.diff_pool import DiffPool
from model.gcn import GCN, GCNWithJK, AFGCN, ConvG
from model.GraUNet import Classifier
from model.edge_pool import EdgePool
from model.gcn import GCN, GCNWithJK
from model.gin import GIN, GIN0, GIN0WithJK, GINWithJK
from model.global_attention import GlobalAttentionNet
from model.graph_sage import GraphSAGE, GraphSAGEWithJK
from model.sag_pool import SAGPool
from model.asap import ASAP
from model.sort_pool import SortPool
from model.GraphGPS import Graphgps
from model.GIUNet import GIUNetCent

import math

from train_eval import cross_validation_with_val_set

#graphU-Net and graU-Net

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--hidden1', type=int, default=128)
parser.add_argument('--hidden2', type=int, default=64)
parser.add_argument('--hidden3', type=int, default=32)
parser.add_argument('--Inc', type=int, default=2)

parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')
'''
#others
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
args = parser.parse_args()
'''

args = parser.parse_args()

layers = [5]
hiddens = [128]
datasets = ['IMDB-MULTI', 'MUTAG', 'IMDB-BINARY', 'REDDIT-BINARY', 'DD', 'NCI1', 'PROTEINS', 'COLLAB']
#datasets = ['COLLAB']
#datasets = ['NCI1']

nets = [
    #SAGPool,
    #DiffPool,
    #EdgePool,
    #GCN,
    #GraphSAGE,
    #GIN,
    #GlobalAttentionNet,
    #SortPool,
    #ASAP,
    #ConvG,
    #Graphgps,
    #GIUNetCent,
    #AFGCN,
    Classifier,
]

#nets = [nets]


def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print(f'{fold:02d}/{epoch:03d}: Val Loss: {val_loss:.4f}, '
          f'Test Accuracy: {test_acc:.4f}')


results = []
for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    print(f'--\n{dataset_name} - {Net.__name__}')
    for num_layers, hidden in product(layers, hiddens):
        dataset = get_dataset(dataset_name, sparse=Net != DiffPool)
        #model = Net(dataset, num_layers, args)

        num_feat = dataset.num_features
        num_classes = dataset.num_classes
        num_node_list = sorted([data.num_nodes for data in dataset])
        s_k = num_node_list[int(math.ceil(0.6 * len(num_node_list))) - 1]
        sk = max(10, s_k)

        #graU-Ne
        model = Classifier(
            node_feat=num_feat,
            nn_hid=48,
            nn_out=97,
            k=sk,
            hidden=128,
            num_class=num_classes
        )


        #others models

        #model = Net(dataset.num_features, dataset.num_classes)


        loss, acc, std = cross_validation_with_val_set(
            dataset,
            model,
            folds=10,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            logger=None,
        )

        if loss < best_result[0]:
            best_result = (loss, acc, std)

    desc = f'{best_result[1]:.4f} ± {best_result[2]:.4f}'
    print(f'Best result - {desc}')
    results += [f'{dataset_name} - {model}: {desc}']
results = '\n'.join(results)
print(f'--\n{results}')
