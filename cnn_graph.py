from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cross_validation import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

####### DATA PREPROCESSING #######

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
test_nodes = int(0.2*13533)   #2706
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
B = np.zeros((len(G.nodes()), (len(H.nodes()) - len(G.nodes()))))
# print B.shape[0], B.shape[1]
train_adjacency_matrix = np.concatenate((train_adjacency_matrix, B), axis = 1)
# print train_adjacency_matrix.shape[0], train_adjacency_matrix.shape[1] 

test_adjacency_matrix = nx.adjacency_matrix(P).todense()
# print test_adjacency_matrix.shape[0]
B = np.zeros((len(P.nodes()), (len(H.nodes()) - len(P.nodes()))))
# print B.shape[0], B.shape[1]
test_adjacency_matrix = np.concatenate((test_adjacency_matrix, B), axis = 1)
# print test_adjacency_matrix.shape[0], test_adjacency_matrix.shape[1] 

d_labels = dict()
for i in range(num_nodes+1):
  d_labels[i] = []
for i,k in zip(dataset_nodes["0"], dataset_nodes["0.1"]):
  d_labels[i] = k

y_train = [d_labels[i] for i in G.nodes()]
y_test = [d_labels[i] for i in P.nodes()]

y_train = np.array(y_train)
y_test = np.array(y_test)

####### DATA PREPROCESSING ENDS #######

####### CNN #######

# Initialising the CNN
classifier = Sequential()

# Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (11422, 11703, 3), activation = 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
# classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN

# classifier.fit_generator(train_adjacency_matrix, steps_per_epoch = 8000, epochs = 25, validation_data = y_train, validation_steps = 2000)