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


### Datasets
We provide the processed dataset and you can run the code directly.

### Running the code

## Node classification on real-world datasets (./NodeClassification)
We evaluate the performance of TSCNet against the competitors on 10 real-world datasets.

### Datasets
We provide the datasets in the folder './NodeClassification/data' and you can run the code directly, or you can choose not to download the datasets('./NodeClassification/data') here. The code will automatically build the datasets through the data loader of Pytorch Geometric.

### Running the code

You can run the following script in the folder './NodeClassification' directly and this script describes the hyperparameters settings of TSCNet on each dataset.
```sh
sh run.sh
```
or run the following Command 
+ Pubmed
```sh
python training_batch.py  --dataset Pubmed --lr 0.01 --dprate 0.5 --dropout 0.0  --train_rate 0.6 --val_rate 0.2 --early_stopping 200
```
+ Texas
```sh
python training_batch.py  --dataset Texas --lr 0.03 --dprate 0.6 --dropout 0.9  --train_rate 0.6 --val_rate 0.2 --early_stopping 100
```

## Citation


## Contact

If you have any questions, please feel free to contact me with z20070009@gmail.com

