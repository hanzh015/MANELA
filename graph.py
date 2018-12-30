#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Graph utilities."""

"""
Updated by Han Zhang for DNELA project
"""

import logging
import sys
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
import random
from random import shuffle
from itertools import product,permutations
from scipy.io import loadmat
from scipy.sparse import issparse
from numpy import random as rd
from numpy import arange as ar
import copy

logger = logging.getLogger("deepwalk")


__author__ = "Bryan Perozzi" + "Han Zhang"
__email__ = "bperozzi@cs.stonybrook.edu"

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

class Graph(defaultdict):
	"""Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""	
	def __init__(self,*args):
		"""
		Update: add constructor function so that we can use copy method of defaultdict
		Author: Han Zhang
		"""
		if args:
			super(Graph,self).__init__(*args)
		else:
			super(Graph, self).__init__(list)

	def nodes(self):
		return self.keys()
	
	
	def adjacency_iter(self):
		return self.iteritems()

	def subgraph(self, nodes={}):
		subgraph = Graph()
		
		for n in nodes:
			if n in self:
				subgraph[n] = [x for x in self[n] if x in nodes]
				
		return subgraph

	def make_undirected(self):
	
		t0 = time()

		for v in self.keys():
			for other in self[v]:
				if v != other:
					self[other].append(v)
		
		t1 = time()
		logger.info('make_directed: added missing edges {}s'.format(t1-t0))

		self.make_consistent()
		return self

	def make_consistent(self):
		t0 = time()
		for k in iterkeys(self):
			self[k] = list(sorted(set(self[k])))
		
		t1 = time()
		logger.info('make_consistent: made consistent in {}s'.format(t1-t0))

		self.remove_self_loops()

		return self

	def remove_self_loops(self):

		removed = 0
		t0 = time()

		for x in self:
			if x in self[x]: 
				self[x].remove(x)
				removed += 1
		
		t1 = time()

		logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
		return self

	def check_self_loops(self):
		for x in self:
			for y in self[x]:
				if x == y:
					return True
		
		return False

	def has_edge(self, v1, v2):
		if v2 in self[v1] or v1 in self[v2]:
			return True
		return False

	def degree(self, nodes=None):
		if isinstance(nodes, Iterable):
			return {v:len(self[v]) for v in nodes}
		else:
			return len(self[nodes])

	def order(self):
		"Returns the number of nodes in the graph"
		return len(self)		

	def number_of_edges(self):
		"Returns the number of nodes in the graph"
		return sum([self.degree(x) for x in self.keys()])/2

	def number_of_nodes(self):
		"Returns the number of nodes in the graph"
		return self.order()
	
	def is_connected(self):
		'''
		Update: return whether the graph is connected
		Author: Han Zhang
		'''
		flags = [1]*self.order()
		visit = []
		visit.append(0)
		while visit:
			for nodes in self[visit.pop(0)]:
				if flags[nodes]==1:
					visit.append(nodes)
					flags[nodes]=0
		
		
		return sum(flags)==0
	

	def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
		""" Returns a truncated random walk.

				path_length: Length of the random walk.
				alpha: probability of restarts.
				start: the start node of the random walk.
		"""
		G = self
		if start:
			path = [start]
		else:
			# Sampling is uniform w.r.t V, and not w.r.t E
			path = [rand.choice(list(G.keys()))]

		while len(path) < path_length:
			cur = path[-1]
			if len(G[cur]) > 0:
				if rand.random() >= alpha:
					path.append(rand.choice(G[cur]))
				else:
					path.append(path[0])
			else:
				break
		return [str(node) for node in path]

# TODO add build_walks in here

def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
											rand=random.Random(0)):
	walks = []

	nodes = list(G.nodes())
	
	for cnt in range(num_paths):
		rand.shuffle(nodes)
		for node in nodes:
			walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
	
	return walks

def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
											rand=random.Random(0)):
	walks = []

	nodes = list(G.nodes())

	for cnt in range(num_paths):
		rand.shuffle(nodes)
		for node in nodes:
			yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)


def clique(size):
		return from_adjlist(permutations(range(1,size+1)))


# http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
def grouper(n, iterable, padvalue=None):
		"grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
		return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def parse_adjacencylist(f):
	adjlist = []
	for l in f:
		if l and l[0] != "#":
			introw = [int(x) for x in l.strip().split()]
			row = [introw[0]]
			row.extend(set(sorted(introw[1:])))
			adjlist.extend([row])
	
	return adjlist

def parse_adjacencylist_unchecked(f):
	adjlist = []
	for l in f:
		if l and l[0] != "#":
			adjlist.extend([[int(x) for x in l.strip().split()]])
	
	return adjlist

def load_adjacencylist(file_, undirected=False, chunksize=10000, unchecked=True):

	if unchecked:
		parse_func = parse_adjacencylist_unchecked
		convert_func = from_adjlist_unchecked
	else:
		parse_func = parse_adjacencylist
		convert_func = from_adjlist

	adjlist = []

	t0 = time()
	
	total = 0 
	with open(file_) as f:
		for idx, adj_chunk in enumerate(map(parse_func, grouper(int(chunksize), f))):
			adjlist.extend(adj_chunk)
			total += len(adj_chunk)
	
	t1 = time()

	logger.info('Parsed {} edges with {} chunks in {}s'.format(total, idx, t1-t0))

	t0 = time()
	G = convert_func(adjlist)
	t1 = time()

	logger.info('Converted edges to graph in {}s'.format(t1-t0))

	if undirected:
		t0 = time()
		G = G.make_undirected()
		t1 = time()
		logger.info('Made graph undirected in {}s'.format(t1-t0))

	return G 


def load_edgelist(file_, undirected=True):
	G = Graph()
	with open(file_) as f:
		for l in f:
			x, y = l.strip().split(",")[:2]
			
			x = int(x)
			y = int(y)
			G[x].append(y)
			if undirected:
				G[y].append(x)
	
	G.make_consistent()
	return G


def load_matfile(file_, variable_name="network", undirected=True):
	mat_varables = loadmat(file_)
	mat_matrix = mat_varables[variable_name]

	return from_numpy(mat_matrix, undirected)


def from_networkx(G_input, undirected=True):
		G = Graph()

		for idx, x in enumerate(G_input.nodes_iter()):
				for y in iterkeys(G_input[x]):
						G[x].append(y)

		if undirected:
				G.make_undirected()

		return G


def from_numpy(x, undirected=True):
		G = Graph()

		if issparse(x):
				cx = x.tocoo()
				for i,j,v in zip(cx.row, cx.col, cx.data):
						G[i].append(j)
		else:
			raise Exception("Dense matrices not yet supported.")

		if undirected:
				G.make_undirected()

		G.make_consistent()
		return G


def from_adjlist(adjlist):
		G = Graph()
		
		for row in adjlist:
				node = row[0]
				neighbors = row[1:]
				G[node] = list(sorted(set(neighbors)))

		return G


def from_adjlist_unchecked(adjlist):
		G = Graph()
		
		for row in adjlist:
				node = row[0]
				neighbors = row[1:]
				G[node] = neighbors

		return G



'''
Updated on Dec 24th 2018, Auther : Han Zhang
adding utils to facilitate evaluations
we assume the graph is always undirected in this case,
so the following methods only consider undirected case
'''
def re_label_nodes(G,dictionary):
	'''
	relabel the nodes using the passing dictionary
	'''
	newgraph = Graph()
	
	for key,value in dictionary.items():
		neighbors = []
		for nodes in G[key]:
			if nodes in dictionary:
				neighbors.append(dictionary[nodes])
			else:
				pass
		newgraph[value] = neighbors
		
	return newgraph
	


def weak_connected_components(G):
	'''
	return a list of connected components, each of them is a graph
	repeatedly running BST 
	'''
	graphset=[]
	original = list(G.keys())
	flags = [1]*len(G)
	
	while(len(original)!=0):
		component = Graph()
		q = [original[0]]
		flags[q[0]]=0
		component[q[0]]=G[q[0]].copy()
		original.remove(q[0])
		while q:
			root = q.pop(0)
			for node in G[root]:
				if flags[node]!=0:
					q.append(node)
					flags[node]=0
					component[node]=G[node]
					original.remove(node)
				else:
					pass
		graphset.append(component)
		
	return graphset
	

def graph_splitter(G, train_ratio):
	'''
	split the graph to train and test
	'''
	train_set = copy.deepcopy(G)
	test_set = copy.deepcopy(G)
	for v_i in range(len(G)):
		for v_j in G[v_i]:
			if v_j<v_i:
				pass
			else:
				if rd.uniform() > train_ratio:
					train_set[v_i].remove(v_j)
					train_set[v_j].remove(v_i)
				else:
					test_set[v_i].remove(v_j)
					test_set[v_j].remove(v_i)
	
	return train_set,test_set

def sample_graph(G,sample_nodes):
	node_num = G.order()
	if sample_nodes < node_num:
		node_l = random.sample(list(G.keys()),sample_nodes)
		node_l = sorted(node_l)
		node_inv = {v:k  for k,v in enumerate(node_l)}
		newgraph = Graph()
		for node in range(len(node_l)):
			newgraph[node]=[]
			for old in G[node_l[node]]:
				try:
					newgraph[node].append(node_inv[old])
				except:
					continue
		return newgraph,node_l
	else:
		#sample nodes larger than # of graph nodes
		return copy.deepcopy(G), ar(node_num)

