import matplotlib.pyplot as plt
from time import time
import networkx as nx
try: import cPickle as pickle
except: import pickle
import numpy as np
import graph
from gensim.models import Word2Vec
import distributed as ds
import argparse

from gem.utils      import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from gem.embedding.gf       import GraphFactorization
from gem.embedding.lle      import LocallyLinearEmbedding
from baseline.node2vec import node2vec

# File that contains the edges. Format: source target
# Optionally, you can add weights as third column: source target weight
path = 'D:\\Project_2018\\Graph_Embedding_Methods\\GEM\\tests\\data\\'
file_prefix = path+'sbm.gpickle'
# Specify whether the edges are directed
isDirected = True

# Load graph
#G = graph_util.loadGraphFromEdgeListTxt(file_prefix, directed=isDirected)
#G = G.to_directed()
G = nx.read_gpickle(file_prefix)

parser = argparse.ArgumentParser()
parser.add_argument('--ratio',type=float,help="MANELA r_1")
args = parser.parse_args()

try:
    f = open(path+'sbm_node_labels.pickle', 'rb')
    node_colors = pickle.load(f)
except UnicodeDecodeError:
    f.seek(0)
    node_colors = pickle.load(f, encoding='latin1')
node_colors_arr = [None] * node_colors.shape[0]
for idx in range(node_colors.shape[0]):
    node_colors_arr[idx] = np.where(node_colors[idx, :].toarray() == 1)[1][0]
    


models = ['manela']

for model in models:
    if model=='deepwalk':
        gr = graph.from_networkx(G,undirected=True)
        walks = graph.build_deepwalk_corpus(gr, 10, 80, 0)
        model = Word2Vec(walks,size=128,window=10,min_count=0, sg=1, hs=0,negative=5, workers=4,iter=1)
        emb_matrix = np.zeros((len(gr),128))
        for key in range(len(gr)):
            emb_matrix[key] = model.wv.get_vector(str(key))
            
    elif model=='manela':
        gr = graph.from_networkx(G,undirected=True)
        emb = ds.Distributed(gr)
        emb.setArgs(numUpdates=90,
                    outputPath='temp_emb.embeddings',
                    representSize=128,
                    window=10,
                    numNegSampling=5,
                    ratio=args.ratio)
        emb.process()
        emb_matrix = emb.getEmbeddings()
        
    elif model=='gf':
        md = GraphFactorization(d=128, max_iter=1000, eta=1 * 10**-4, regu=1.0, data_set='sbm')
        md.learn_embedding(G)
        emb_matrix = md.get_embedding()
    
    elif model=='node2vec':
        gra = node2vec.Graph(G,is_directed=isDirected,p=1,q=1)
        gra.preprocess_transition_probs()
        walks = gra.simulate_walks(num_walks=10, walk_length=80)
        walks = [list(map(str, walk)) for walk in walks]
        md=Word2Vec(walks, size=128, window=10, min_count=0, sg=1, workers=4, iter=1)
        emb_matrix = np.zeros((1024,128))
        for key in range(1024):
            emb_matrix[key] = md.wv.get_vector(str(key))
    elif model=='lle':
        md = LocallyLinearEmbedding(d=128)
        md.learn_embedding(G)
        emb_matrix = md.get_embedding()
        
    else:
        pass
    
    viz.plot_embedding2D(emb_matrix, di_graph=G, node_colors=node_colors_arr)
    plt.show()
    plt.clf()
        