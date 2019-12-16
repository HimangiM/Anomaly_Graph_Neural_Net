# 14855843 edges
# anomaly detection datasets social networks

import pandas as pd
import networkx as nx

dataset_graph = pd.read_csv('higgs-social_network.edgelist', sep= " ", header = None)

node_A = dataset_graph.iloc[:,0].values
node_B = dataset_graph.iloc[:,1].values

# print node_A


G = nx.Graph()

for i, j in zip(node_A, node_B):
	G.add_edge(i, j)


# nx.write_graphml(G, 'twitter.graphml')
print len(G.edges()), len(G.nodes())





