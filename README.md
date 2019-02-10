# MANELA: A Multi-Agent Algorithm for Learning Network Embeddings

This repo provides an reference implementation of MANELA simulator. For detailed informations, see.

## Organization
This folder is organized as follows. 
* baseline\\:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Implementations of baseline methods, including node2vec and relationalneighbors
* examples\\: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;        Example datasets.
* \_\_main\_\_.py:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The entrance script for training MANELA embeddings. Specify hyperparameters here and it will call distributed.py.
* distributed.py:  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  The main implementation module for MANELA simulator.
* edge_prediction.py: The module for validating link prediction performance.
* evaluation_utils.py: A module providing necessary utilities for evaluations.
* graph.py:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Implementation of networks and their corresponding operations.
* scoring.py: A module for validating node classification performance.
* visualize.py: A module implemented network visualizations.

## Usage
### Basic: Training MANELA embeddings.
A simple example to run \_\_main__.py is as follows. It takes BlogCatalog network as input file, trains embeddings using MANELA algorithm, and save blogcatalog.embeddings as output file. The input file is assumed to be in .mat format, whereas output files follow the format in Word2vec model of gensim.

    python __main__.py --path examples\datasets\blogcatalog.mat --output examples\embeddings\blogcatalog.embeddings --updates 60 --ratio 0.3 --window 10
The optional arguments specify hyperparameters of MANELA algorithm:
> * **--window**: The window size hyperparameter, denoted as w in the paper.
> * **--ratio**: The r<sub>1</sub> component of the ratio vector, where we assume ratio vector is two dimensional
> * **--updates**: The update hyperparameter unique in MANELA simulator, where definitions are: in one unit update, each node v performs d(v)/w iterations. d(v) is the degree of node v. A reasonable rounding mechanism is performed here to guarantee interger iteration numbers.
> * **--dimension**: The dimension of embedding vectors, denoted as d in the paper. Set to 128 by default.
> * **--negative**: The number of negative samples in between two positive updates, denoted as k in the paper. Set to 5 by default.
> * **--alpha**: The initial learning rate. Set to 0.025 by default.
> * **--seed**: Random seed, 0 by default.
### Validations: Testing performance in various tasks.
#### Node Classification
To score the node classification, run:

    python scoring.py --emb examples\embeddings\blogcatalog.embeddings --network examples\datasets\blogcatalog.mat --result examples\results\blogcatalog.txt --all --num-shuffles 10
    
The arguments of scoring module:
> * **--emb**: The embedding path.
> * **--network**: The path of the network dataset.
> * **--num-shuffles**: The number of runs that f1 scores are averaged over.
> * **--result**: The path of the report file. By default it will only display results on the screen without saving them to files.

#### Link Prediction



