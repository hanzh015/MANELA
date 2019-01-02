#!/usr/bin/env python

"""scoring.py: Script that demonstrates the multi-label classification used."""
"""
updated on Dec 31 2018
author : Han Zhang
adding baseline method RN for comparing results
"""

__author__			= "Bryan Perozzi" + "Han Zhang"

import numpy
import sys

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict
from gensim.models import Word2Vec, KeyedVectors
from six import iteritems
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer

class TopKRanker(OneVsRestClassifier):
	def predict(self, X, top_k_list):
		assert X.shape[0] == len(top_k_list)
		probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
		all_labels = []
		for i, k in enumerate(top_k_list):
			probs_ = probs[i, :]
			labels = self.classes_[probs_.argsort()[-k:]].tolist()
			all_labels.append(labels)
		return all_labels

def sparse2graph(x):
	G = defaultdict(lambda: set())
	cx = x.tocoo()
	for i,j,_ in zip(cx.row, cx.col, cx.data):
		G[i].add(j)
	#modified: change str(k) to k
	return {k: [x for x in v] for k,v in iteritems(G)}

def sparse2adjlist(x):
	graph = [[]] * x.shape[0]
	cx = x.tocoo()
	for i,j,_ in zip(cx.row,cx.col,cx.data):
		graph[i].append(j)
	return graph

def rnPredict(x_fix,x_iter,label_fix,label_iter,graph):
	'''
	RN baseline method for node prediction task
	input: 
	output: all_labels : the finished label matrix for such method
	'''
	#1.map nodes key to its labels
	label_map = defaultdict(list)
	for k, node in enumerate(x_fix):
		label_map[node] = numpy.squeeze(numpy.asarray(label_fix[k]))
	for k, node in enumerate(x_iter):
		label_map[node] = numpy.squeeze(numpy.asarray(label_iter[k]))	
			
	label_count = 0
	max_count = label_iter.shape[0] * label_iter.shape[1]
	thred = 0
	while label_count<max_count:
		#voting process
		flag = True
		for node in x_iter:
			vote={k:[0]*label_fix.shape[1] for k in [-1,0,1]}
			for neighborhood in graph[node]:
				for labels in range(label_fix.shape[1]):
					if label_map[neighborhood][labels]==-1:
						vote[-1][labels] += 1
					elif label_map[neighborhood][labels]==0:
						vote[0][labels] += 1
					elif label_map[neighborhood][labels]==1:
						vote[1][labels] += 1
			for label in range(label_fix.shape[1]):
				label_rank = sorted(vote.items(),key=lambda x:x[1][label])
				if label_rank[-1][0] == -1:
					if label_rank[-1][1][label]>thred*(label_rank[0][1][label]+label_rank[1][1][label]+label_rank[2][1][label]):
						new_label = -1
					else:
						new_label = label_rank[-2][0]
				else:
					new_label = label_rank[-1][0]
				if label_map[node][label]==-1 and new_label!=-1:
					label_count += 1
					flag=False
				label_map[node][label] = new_label
		if flag:
			thred += 0.1
		print("progress: {}% {}/{}".format(100*float(label_count)/max_count,label_count,max_count))	
	result = numpy.zeros((len(x_iter),label_fix.shape[1]))
	for k, node in enumerate(x_iter):
		result[k] = label_map[node]
		
	return result
	

def main():
	parser = ArgumentParser("scoring",
													formatter_class=ArgumentDefaultsHelpFormatter,
													conflict_handler='resolve')
	parser.add_argument("--emb", required=True, help='Embeddings file')
	parser.add_argument("--network", required=True,
											help='A .mat file containing the adjacency matrix and node labels of the input network.')
	parser.add_argument("--adj-matrix-name", default='network',
											help='Variable name of the adjacency matrix inside the .mat file.')
	parser.add_argument("--label-matrix-name", default='group',
											help='Variable name of the labels matrix inside the .mat file.')
	parser.add_argument("--num-shuffles", default=2, type=int, help='Number of shuffles.')
	parser.add_argument("--all", default=False, action='store_true',
											help='The embeddings are evaluated on all training percents from 10 to 90 when this flag is set to true. '
											'By default, only training percents of 10, 50 and 90 are used.')
	parser.add_argument("--result", required=True, help="Result file")
	parser.add_argument("-b",action='store_true',help='whether to use baseline method metric')

	args = parser.parse_args()
	# 0. Files
	embeddings_file = args.emb
	matfile = args.network
	
	# 1. Load Embeddings
	model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
	
	# 2. Load labels
	mat = loadmat(matfile)
	A = mat[args.adj_matrix_name]
	graph = sparse2graph(A)
	keys = list(range(len(graph)))
	labels_matrix = mat[args.label_matrix_name]
	labels_count = labels_matrix.shape[1]
	mlb = MultiLabelBinarizer(range(labels_count))
	
	# Map nodes to their features (note:	assumes nodes are labeled as integers 1:N)
	features_matrix = numpy.asarray([model[str(node)] for node in range(len(graph))])
	
	# 2. Shuffle, to create train/test groups
	shuffles = []
	for _ in range(args.num_shuffles):
		shuffles.append(skshuffle(features_matrix, labels_matrix,keys))
	
	# 3. to score each train/test group
	all_results = defaultdict(list)
	baseline_results = defaultdict(list)
	
	if args.all:
		training_percents = numpy.asarray(range(1, 5)) * .1
	else:
		training_percents = [0.1, 0.5, 0.9]
	for train_percent in training_percents:
		for shuf in shuffles:
	
			X, y, adj = shuf
	
			training_size = int(train_percent * X.shape[0])
			testing_size = X.shape[0]-training_size
			
			
			X_train = X[:training_size, :]
			y_train_ = y[:training_size]
			node_fix = adj[:training_size]
			node_iter = adj[training_size:]
			test_label_matrix = numpy.array([[-1]*labels_matrix.shape[1]]*testing_size)
	
			y_train = [[] for _ in range(y_train_.shape[0])]
	
	
			cy =	y_train_.tocoo()
			for i, j in zip(cy.row, cy.col):
					y_train[i].append(j)
	
			assert sum(len(l) for l in y_train) == y_train_.nnz
	
			X_test = X[training_size:, :]
			y_test_ = y[training_size:]
	
			y_test = [[] for _ in range(y_test_.shape[0])]
	
			cy =	y_test_.tocoo()
			for i, j in zip(cy.row, cy.col):
					y_test[i].append(j)
	
			clf = TopKRanker(LogisticRegression(solver='liblinear'))
			clf.fit(X_train, y_train_)
	
			# find out how many labels should be predicted
			top_k_list = [len(l) for l in y_test]
			preds = clf.predict(X_test, top_k_list)
			
			#get baseline results
			if args.b:
				preds_base = rnPredict(node_fix,node_iter,y_train_.tocoo().todense(),numpy.array(test_label_matrix),graph)
	
			results = {}
			bresults = {}
			averages = ["micro", "macro"]
			for average in averages:
				results[average] = f1_score(mlb.fit_transform(y_test), mlb.fit_transform(preds), average=average)
				if args.b:
					bresults[average] = f1_score(mlb.fit_transform(y_test),preds_base,average=average)
	
			all_results[train_percent].append(results)
			if args.b:
				baseline_results[train_percent].append(bresults)
	
	try:
		file = open(args.result,'w')
		file.write('Results, using embeddings of dimensionality'+str(X.shape[1])+'\n')
		file.write('-------------------\n')
		print ('Results, using embeddings of dimensionality', X.shape[1])
		print ('-------------------')
		for train_percent in sorted(all_results.keys()):
			print ('Train percent:', train_percent)
			file.write('Train percent:'+str(train_percent)+'\n')
			for index, result in enumerate(all_results[train_percent]):
				print ('Shuffle #%d:	 ' % (index + 1), result)
				file.write('Shuffle #{}:     '.format(index+1)+str(result)+'\n')
			avg_score = defaultdict(float)
			for score_dict in all_results[train_percent]:
				for metric, score in iteritems(score_dict):
					avg_score[metric] += score
			for metric in avg_score:
				avg_score[metric] /= len(all_results[train_percent])
			print ('Average score:', dict(avg_score))
			file.write('Average score:' + str(dict(avg_score))+'\n')
			print ('-------------------')
			file.write('-------------------\n')
			
		#write baseline method result for comparison
		if args.b:
			file.write('Results, using embeddings of dimensionality'+str(X.shape[1])+'\n')
			file.write('-------------------\n')
			for train_percent in sorted(baseline_results.keys()):
				file.write('Train percent:'+str(train_percent)+'\n')
				for index, result in enumerate(baseline_results[train_percent]):
					file.write('Shuffle #{}:     '.format(index+1)+str(result)+'\n')
				avg_score = defaultdict(float)
				for score_dict in baseline_results[train_percent]:
					for metric, score in iteritems(score_dict):
						avg_score[metric] += score
				for metric in avg_score:
					avg_score[metric] /= len(baseline_results[train_percent])
				file.write('Average score:' + str(dict(avg_score))+'\n')
				file.write('-------------------\n')
				
	finally:
		file.close()

if __name__ == "__main__":
	sys.exit(main())
