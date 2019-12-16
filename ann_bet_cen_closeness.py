import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cross_validation import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import csv

dataset_edges = pd.read_csv('enrondatasetfinal.csv', sep=',')
# dataset_nodes = pd.read_csv('bet_cen_enron_data.csv', sep = ' ', header = None, quoting = csv.QUOTE_NONE, error_bad_lines = False)
dataset_true = pd.read_csv('Enron.true', header = None, sep = ';')

node_from = dataset_edges.iloc[:,0].values
node_to = dataset_edges.iloc[:,1].values

num_nodes = len(dataset_true)
print "Number of nodes: " +  str(num_nodes)

#Stores the node as key, it's neighbors as values
d = dict()
for i in range(0, num_nodes):
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

print len(H.nodes())
# print len(H.edges())
print len(G.nodes())
print len(P.nodes())

train_adjacency_matrix = nx.adjacency_matrix(G).todense()
# print train_adjacency_matrix.shape[0], train_adjacency_matrix.shape[1]
# np zero is added to show the if there is an edge between all the other nodes
B = np.zeros((len(G.nodes()), (len(H.nodes()) - len(G.nodes()))))
# print B.shape[0], B.shape[1]
train_adjacency_matrix = np.concatenate((train_adjacency_matrix, B), axis = 1)
print train_adjacency_matrix.shape[0], train_adjacency_matrix.shape[1] 

test_adjacency_matrix = nx.adjacency_matrix(P).todense()
# print test_adjacency_matrix.shape[0]
B = np.zeros((len(P.nodes()), (len(H.nodes()) - len(P.nodes()))))
# print B.shape[0], B.shape[1]
test_adjacency_matrix = np.concatenate((test_adjacency_matrix, B), axis = 1)
print test_adjacency_matrix.shape[0], test_adjacency_matrix.shape[1] 

d_labels = dict()
for i in range(0, num_nodes):
	d_labels[i] = []
with open('bet_cen_closeness_enron_data.txt', 'r') as infile:
    for line in infile.readlines():
        token = line.strip().split(" ")
	d_labels[int(token[0])].append(int(token[1]))


# print d_labels

y_train = [v for k,v in d_labels.iteritems() if k in G.nodes()]
y_test = [d_labels[i] for i in P.nodes()]
# print y_train[10]

y_train = np.array(y_train)
y_test = np.array(y_test)
classifier = Sequential()


#First hidden layer
classifier.add(Dense(output_dim = 5711, init = 'uniform', activation = 'relu', input_dim = 11703))

#Second hidden layer
# classifier.add(Dense(units = 11703, kernel_initializer = 'uniform', activation = 'relu'))

#Output Layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compiling model
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Training
classifier.fit(train_adjacency_matrix, y_train, batch_size = 10, epochs = 100)

scores = model.evaluate(train_adjacency_matrix, y_train)
print "Train scores: " + str(classifier.metrics_names[1]) + str(scores[1]*100)
#Prediction
y_pred = classifier.predict(test_adjacency_matrix)
y_pred = (y_pred > 0.5)

scores = model.evaluate(test_adjacency_matrix, y_test)
print "Train scores: " + str(classifier.metrics_names[1]) + str(scores[1]*100)

