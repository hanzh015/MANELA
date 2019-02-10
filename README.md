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
### Basic: Training MANELA embeddings
    python __main__.py --path examples\datasets\blogcatalog.mat --output examples\embeddings\blogcatalog.embeddings --updates 60 --ratio 0.3 --window 10
