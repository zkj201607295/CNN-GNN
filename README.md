# CNN-GNN
CNN-GNN Convolution Bridge: An effective algorithmic migration strategy from CNNs to GNNs

This repository contains a PyTorch implementation of "CNN-GNN Convolution Bridge: An effective algorithmic migration strategy from CNNs to GNNs".

## Environment Settings    
- pytorch 1.8.1
- numpy 1.18.1
- torch-geometric 2.3.1 
- tqdm 4.59.0
- scipy 1.6.2
- seaborn 0.11.1
- scikit-learn 0.24.1

### Running the code

## Node classification on real-world datasets (./NodeClassification)
We evaluate the performance of GraInc against the competitors on 9 real-world datasets.(Cora, Citeseer, PubMed, Computers, Photo, Actor, Texas, Cornell, OGBN-Arxiv)

### Datasets
We provide the datasets in the folder './NodeClassification/data' and you can run the code directly, or you can choose not to download the datasets('./NodeClassification/data') here. The code will automatically build the datasets through the data loader of Pytorch Geometric.

### Running the code

You can run the following script in the folder './NodeClassification' directly and this script describes the hyperparameters settings of GraInc on each dataset.
+ Pubmed
```sh
python training.py --net=GraInc  --dataset Photo --lr 0.01 --dprate 0.5 --dropout 0.5  --train_rate 0.6 --val_rate 0.2 --early_stopping 200
```
+ Actor
```sh
python training.py --net=GraInc  --dataset Actor --lr 0.01 --dprate 0.9 --dropout 0.5  --train_rate 0.6 --val_rate 0.2 --early_stopping 100
```
+ OGBN-Arxiv
```sh
python training.py --net=GraInc_Arxiv  --dataset ogbn-arxiv --lr 0.002 --dprate 0.5 --dropout 0.1 --early_stopping 600
```

## Graph classification on real-world datasets (./NodeClassification)
We evaluate the performance of GraU-Net against the competitors on 8 real-world datasets.(DD, PROTEINS, NCI1, MUTAG, COLLAB, REDDIT-IMDB, IMDB-BINARY, IMDB-MULTI)
    - 


### Datasets
We provide the datasets in the folder './GraphClassification/data' and you can run the code directly, or you can choose not to download the datasets('./GraphClassification/data') here. The code will automatically build the datasets through the data loader of Pytorch Geometric.

### Running the code

You can run the following script in the folder './GraphClassification' directly and all the hyperparameters settings of GraphU-Net on each dataset are the same.
+ All datasets
```sh
python main.py
```

## Citation

## Contact

If you have any questions, please feel free to contact me with z20070009@gmail.com

