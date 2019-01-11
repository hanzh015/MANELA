import numpy
from collections import defaultdict
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