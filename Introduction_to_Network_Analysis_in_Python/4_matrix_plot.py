# Import nxviz
import nxviz as nv
import networkx as nx
from nxviz import matrix
import pickle
import matplotlib.pyplot as plt

# Create the matrix plot: m
T = pickle.load(open("Data\ego-twitter.p", 'rb'))
m = matrix(T)

# Display the plot
plt.show()

# Convert T to a matrix format: A
A = nx.to_numpy_matrix(T)

# Convert A back to the NetworkX form as a directed graph: T_conv
T_conv = nx.from_numpy_matrix(A, create_using=nx.DiGraph())

# Check that the `category` metadata field is lost from each node
for n, d in T_conv.nodes(data=True):
    assert 'category' not in d.keys()