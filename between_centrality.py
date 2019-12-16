import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cross_validation import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

dataset_edges = pd.read_csv('./Enron/enron_dataset_final.csv')
dataset_nodes = pd.read_csv('./Enron/Enron.true', sep = ';')

node_from = dataset_edges.iloc[:,0].values
node_to = dataset_edges.iloc[:,1].values

num_nodes = len(dataset_nodes)
print "Number of nodes: " +  str(num_nodes)

#Stores the node as key, it's neighbors as values
d = dict()
for i in range(0, (num_nodes+1)):
	d[i] = []
for (i,j) in zip(node_from, node_to):
	if j not in d[i]:
		d[i].append(j)
	if i not in d[j]:
		d[j].append(i)

train_nodes = int(0.8*13533)    #10826 
test_nodes = int(0.2*13533) 	#2706
print 'Train nodes: ' + str(train_nodes) + ", Test nodes: " + str(test_nodes)

#Full dataset graph
H = nx.Graph()
for k, v in d.iteritems():
	for j in v:
		H.add_edge(k, j)

# Training graph
G = nx.Graph()
for i in range(train_nodes):
	for j in d[i]:
		G.add_edge(i, j)

#Testing graph
test_id = [i for i in H.nodes() if i not in G.nodes()]
P = nx.Graph()
for i in test_id:
	for j in d[i]:
		P.add_edge(i, j)

print nx.betweenness_centrality(G)