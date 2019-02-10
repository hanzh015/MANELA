'''
Dec 24th, 2018, author :Han Zhang
'''
from numpy import zeros,dot
from sklearn.metrics import roc_auc_score

def get_recontructed_adj(embeddings):
    node_num = len(embeddings)
    adj_matrix = zeros((node_num,node_num))
    for v_i in range(node_num):
        for v_j in range(node_num):
            if v_i==v_j:
                continue
            else:
                adj_matrix[v_i][v_j]=dot(embeddings[v_i],embeddings[v_j])
    return adj_matrix
    

def get_edge_list_from_adj(adj, threshold=0.0, edge_pairs=None):
    result = []
    node_num = adj.shape[0]
    if edge_pairs:
        for (i,j) in edge_pairs:
            if adj[i][j]>threshold:
                result.append((i,j,adj[i][j]))
    else:
        for v_i in range(node_num):
            for v_j in range(node_num):
                if v_i>=v_j:
                    continue
                else:
                    if adj[v_i][v_j]>threshold:
                        result.append((v_i,v_j,adj[v_i][v_j]))
    
    return result

def get_edge_list_from_cn(node_l, ori_graph, threshold=0.0):
    #construct the score table
    result = []
    for v_i in range(len(node_l)):
        for v_j in range(len(node_l)):
            if v_i >= v_j:
                continue
            else:
                node_i = node_l[v_i]
                node_j = node_l[v_j]
                union_neighbors = ori_graph[node_i]+ori_graph[node_j]
                common_number = len(ori_graph[node_i]+ori_graph[node_j])-len(sorted(set(union_neighbors)))
                if common_number > threshold:
                    result.append((v_i,v_j,common_number))
    
    return result

def compute_precision_curves(predicted_edge_list, true_graph, max_k=-1, a=False):
    if max_k==-1:
        max_k = len(predicted_edge_list)
    else:
        max_k = min(max_k, len(predicted_edge_list))
    sorted_edges = sorted(predicted_edge_list, key=lambda x: x[2], reverse=True)

    precision_scores = []
    delta_factors = []
    correct_edge = 0
    for i in range(max_k):
        if true_graph.has_edge(sorted_edges[i][0], sorted_edges[i][1]):
            correct_edge += 1
            delta_factors.append(1.0)
        else:
            delta_factors.append(0.0)
        precision_scores.append(1.0 * correct_edge / (i + 1))
        
    '''
    update Jan 17th, 2019
    add roc_auc score
    '''
    if a:
        auc = roc_auc_score(delta_factors, precision_scores)
        return precision_scores, delta_factors, auc
    else:
        return precision_scores, delta_factors
    

def compute_map(predicted_edge_list, true_digraph, max_k=-1):
    node_num = true_digraph.number_of_nodes()
    node_edges = []
    for i in range(node_num):
        node_edges.append([])
    for (st, ed, w) in predicted_edge_list:
        node_edges[st].append((st, ed, w))
        node_edges[ed].append((ed, st, w))
    node_AP = [0.0] * node_num
    count = 0
    for i in range(node_num):
        if true_digraph.degree(i) == 0:
            continue
        count += 1
        precision_scores, delta_factors = compute_precision_curves(node_edges[i], true_digraph, max_k)
        precision_rectified = [p * d for p,d in zip(precision_scores,delta_factors)]
        if(sum(delta_factors) == 0):
            node_AP[i] = 0
        else:
            node_AP[i] = float(sum(precision_rectified) / sum(delta_factors))
    return sum(node_AP) / count
    



