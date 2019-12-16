import pandas as pd
import numpy as np
import networkx as nx
import pickle

dataset_edges = pd.read_csv('enrondatasetfinal.csv')
dataset_nodes = pd.read_csv('Enron.true', sep = ';')

node_from = dataset_edges.iloc[:,0].values
node_to = dataset_edges.iloc[:,1].values

num_nodes = len(dataset_nodes)

print num_nodes

# Creare graph
G = nx.Graph()
for i, j in zip(node_from, node_to):
	G.add_edge(i, j)

'''
d_bet_cen = dict()
d_bet_cen = nx.betweenness_centrality(G)
d_degree = G.degree(G.nodes())

output = open('bet_cen_dict.pkl', 'wb')
pickle.dump(d_bet_cen, output)
output.close()

'''
#Reading between central dictionary pickle file
pkl_file = open('bet_cen_dict.pkl','rb')
d_bet_cen =pickle.load(pkl_file)

#Reading degree dictionary pickle file
pkl_file2 = open('degree_list_tup.pkl', 'rb')
d_degree = pickle.load(pkl_file2)

# l = filter(lambda x:x[1]>=0.0000001,d_bet_cen.items())
# print len(l)
# print min(d_bet_cen.items(), key = lambda x:x[1])

bet_nodes = []
for k1, v1 in d_bet_cen.iteritems():
    if v1>0.0000001:
        bet_nodes.append(k1)

cnt = 0
outfile = open('bet_cen_degree_enron_data.txt', 'w')
for item in d_degree:
    if item[0] in bet_nodes and item[1]>70:
        outfile.write(str(item[0]) + " 1\n")
        cnt += 1
    else:
        outfile.write(str(item[0]) + " 0\n")

print cnt
outfile.close()
pkl_file.close()
pkl_file2.close()
