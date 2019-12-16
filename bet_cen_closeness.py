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

output = open('bet_cen_dict.pkl', 'wb')
pickle.dump(d_bet_cen, output)
output.close()

'''
pkl_file = open('bet_cen_dict.pkl','rb')
d_bet_cen =pickle.load(pkl_file)

pkl_file2 = open('closeness_dict.pkl', 'rb')
d_closeness = pickle.load(pkl_file2)

# l = filter(lambda x:x[1]>=0.0000001,d_bet_cen.items())
# print len(l)

# print min(d_bet_cen.items(), key = lambda x:x[1])

# print d_bet_cen

cnt = 0
outfile = open('bet_cen_closeness_enron_data.txt', 'w')
for (k1, v1), (k2, v2) in zip(d_bet_cen.iteritems(), d_closeness.iteritems()):
    if v1 > 0.0000001 and v2>0.25 and k1 == k2:
        cnt += 1
        outfile.write(str(i) + " 1\n")
    else:
        outfile.write(str(i) + " 0\n")
        
print cnt

outfile.close()
pkl_file.close()
pkl_file2.close()
