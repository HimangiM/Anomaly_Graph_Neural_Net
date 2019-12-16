import pandas as pd
import numpy as np
import networkx as nx
import pickle

'''
dataset_edges = pd.read_csv('higgs-social_network.edgelist', sep = " ", header = None)
# dataset_nodes = pd.read_csv('Enron.true', sep = ';')

node_from = dataset_edges.iloc[:,0].values
node_to = dataset_edges.iloc[:,1].values

num_nodes = 76851

print num_nodes

# Creare graph
G = nx.Graph()
for i, j in zip(node_from, node_to):
	G.add_edge(i, j)

'''
num_nodes = 76851
G = nx.read_graphml('twitter.graphml')
print num_nodes
d_bet_cen = dict()
d_bet_cen = nx.betweenness_centrality(G)

print 'Dict done'
output = open('twitter_bet_cen_dict.pkl', 'wb')
pickle.dump(d_bet_cen, output)
output.close()

'''
pkl_file = open('bet_cen_dict.pkl','rb')
d_bet_cen =pickle.load(pkl_file)

l = filter(lambda x:x[1]>=0.0000001,d_bet_cen.items())
print len(l)

# print min(d_bet_cen.items(), key = lambda x:x[1])

# print d_bet_cen

outfile = open('bet_cen_enron_data.txt', 'w')
for i, j in d_bet_cen.iteritems():
    if j >= 0.0000001:
        outfile.write(str(i) + " 1\n")
    else:
        outfile.write(str(i) + " 0\n")

outfile.close()
pkl_file.close()
'''
