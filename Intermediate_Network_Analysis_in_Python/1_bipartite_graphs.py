import networkx as nx
import nxviz as  nv
import matplotlib.pyplot as plt

G = nx.Graph()

numbers = range(3)

G.add_nodes_from(numbers, bipartite = 'customers')

letters = ['a', 'b']

G.add_nodes_from(letters, bipartite = 'products')

print(list(G.nodes(data=True)))

