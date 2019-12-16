import pandas as pd
import networkx as nx

data = pd.read_csv('higgs-social_network.edgelist', header = None, sep = " ")

data_a = data.iloc[:,0].values
data_b = data.iloc[:,1].values

nodes = []
for i, j in zip(data_a, data_b):
    if i not in nodes:
        nodes.append(i)
    if j not in nodes:
        nodes.append(j)

print len(nodes)
