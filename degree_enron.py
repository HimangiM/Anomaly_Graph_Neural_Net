import pandas as pd
import numpy as np
import networkx as nx
import pickle

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

d = G.degree(G.nodes())

# print d

max_ = 0
ind = 0
for a, b in d:
    if b > max_:
        max_ = b
        ind = a

print max_, ind

'''
output = open('degree_list_tup.pkl', 'wb')
pickle.dump(d, output)
output.close()
'''

pkl_file = open('degree_list_tup.pkl','rb')
d = pickle.load(pkl_file)

cnt = [item for item in d if item[1]>70]
print len(cnt)


outfile = open('degree_enron_data.txt', 'w')
for i, j in d:
    if j > 70:
        outfile.write(str(i) + " 1\n")
    else:
        outfile.write(str(i) + " 0\n")


outfile.close()
pkl_file.close()


