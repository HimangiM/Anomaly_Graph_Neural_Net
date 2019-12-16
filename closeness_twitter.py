import pandas as pd
import numpy as np
import networkx as nx
import pickle

'''
dataset_edges = pd.read_csv('enrondatasetfinal.csv')
dataset_nodes = pd.read_csv('Enron.true', sep = ';', header = None)

node_from = dataset_edges.iloc[:,0].values
node_to = dataset_edges.iloc[:,1].values

num_nodes = len(dataset_nodes)

print num_nodes

# Creare graph
G = nx.Graph()
for i, j in zip(node_from, node_to):
	G.add_edge(i, j)

'''
num_nodes = 76851
G = nx.read_graphml('twitter.graphml')
d = nx.closeness_centrality(G) 

print max(d.items(), key = lambda x:x[1])
print min(d.items(), key = lambda x:x[1])

print 'Dict done'
output = open('closeness_dict.pkl', 'wb')
pickle.dump(d, output)
output.close()

'''
pkl_file = open('closeness_dict.pkl','rb')
d = pickle.load(pkl_file)

cnt = 0
for a, b in d.iteritems():
    if b>0.25:
        cnt += 1
print cnt


outfile = open('closeness_enron_data.txt', 'w')
for i, j in d.iteritems():
    if j > 0.25:
        outfile.write(str(i) + " 1\n")
    else:
        outfile.write(str(i) + " 0\n")


outfile.close()
pkl_file.close()

'''
