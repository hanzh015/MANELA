'''
this file score the performance for edge prediction task
Dec 25th, 2018 Author: Han Zhang
'''
import evaluation_utils as eu
import graph
import distributed as ds
from gensim.models import Word2Vec
from numpy import zeros
import baseline.node2vec.node2vec as n2v
import networkx as nx

def evaluatePrediction(ori_graph,emb_name='dnela',train_ratio=0.8,sample_nodes=None):
    #1. split the original graph to train and test. Remove edges from original graph 
    #to create train graph, the complimentary part left is test graph
    #if the split train graph is not connected, return the max connected component
    print(ori_graph.order())
    print(str(ori_graph.is_connected()))
    train_graph,test_graph = graph.graph_splitter(ori_graph, train_ratio)
    
    if not train_graph.is_connected():
        train_graph = max(graph.weak_connected_components(train_graph),key=len)
        train_nodes = list(train_graph.keys())
        train_nodes_dict = dict(zip(train_nodes,range(len(train_nodes))))
        train_graph = graph.re_label_nodes(train_graph, train_nodes_dict)
        test_graph = test_graph.subgraph(train_nodes)
        test_graph = graph.re_label_nodes(test_graph, train_nodes_dict)
    node_num = train_graph.order()
    print(node_num)
        
    #2. train embeddings using methods specified
    if emb_name=='dnela':
        emb = ds.Distributed(train_graph)
        emb.setArgs(numUpdates=200,
                   outputPath='temp_emb.embeddings',
                   ratio=1)
        emb.process()
        emb_matrix = emb.getEmbeddings()
    elif emb_name=='deepwalk':
        walks = graph.build_deepwalk_corpus(train_graph, 10, 80, 0)
        model = Word2Vec(walks,size=128,window=10,min_count=0, sg=1, hs=0,negative=5, workers=4)
        emb_matrix = zeros((node_num,128))
        for key in range(node_num):
            emb_matrix[key] = model.wv.get_vector(str(key))
        
    elif emb_name=='node2vec':
        #1. transform graph format from graph to nx.Graph()
        ngraph = nx.Graph()
        for key, value in train_graph.items():
            for adj in value:
                ngraph.add_edge(key,adj,weight=1)
        ngraph.to_undirected()
        G = n2v.Graph(ngraph,False,1,1)
        G.preprocess_transition_probs()
        walks = G.simulate_walks(10, 80)
        walks = [list(map(str,walk)) for walk in walks]
        model = Word2Vec(walks,size=128,window=10,min_count=0,sg=1,hs=0,negative=5,workers=4,iter=1)
        emb_matrix = zeros((node_num,128))
        for key in range(node_num):
            emb_matrix[key] = model.wv.get_vector(str(key))
    
    else:
        pass
    
    #3. sample some nodes for validation
    if sample_nodes:
        if sample_nodes<node_num:
            test_graph,node_l = graph.sample_graph(test_graph, sample_nodes)
            emb_matrix = emb_matrix[node_l]
            
    #4. construct node weights from embeddings
    adj_matrix = eu.get_recontructed_adj(emb_matrix)
    result_pair_list = eu.get_edge_list_from_adj(adj_matrix)
    #filter the result edge list from those appeared in train_graph
    #NOTE: THIS STEP IS IMPORTANT SINCE train_set HERE IS COMPLETE, NOT SAMPLED WHILE test_graph
    #IS SAMPLED SO THEY HAVE DIFFRENT LABELS. THIS DICTIONARY IS FOR NODE TRANSLATION
    filtered_pair_list = [pair for pair in result_pair_list if not train_graph.has_edge(node_l[pair[0]],node_l[pair[1]])]
    
    #5. compute MAP and precision curve
    MAP = eu.compute_map(filtered_pair_list, test_graph,max_k=1000)
    precision_curve,_ = eu.compute_precision_curves(filtered_pair_list, test_graph,max_k=1000)
    
    return MAP, precision_curve

def main():
    num_shuffle = 1
    report_file_name = 'examples\\results\\link_pred_node2vec.txt'
    graph_name = 'examples\\datasets\\Homo_sapiens.mat'
    ori_graph = graph.load_matfile(file_=graph_name)
    ori_graph.make_undirected()
    emb_name = 'node2vec'
    train_ratio = 0.8
    sample_node = 1024
    map_round = [None]*num_shuffle
    curve_round = [None]*num_shuffle
    print('start validating link prediction...')
    try:
        file = open(report_file_name,'w')
        for round_id in range(num_shuffle):
            map_round[round_id], curve_round[round_id]=evaluatePrediction(ori_graph, emb_name, train_ratio, sample_node)
            file.write(str(map_round[round_id]))
            print('MAP:{}'.format(map_round[round_id]))
            for i in curve_round[round_id]:
                file.write(" {}".format(i))
            file.write("\n")
    finally:
        file.close()
    
    print("saved to file: {}".format(report_file_name))


if __name__=='__main__':
    main()
    
    
    