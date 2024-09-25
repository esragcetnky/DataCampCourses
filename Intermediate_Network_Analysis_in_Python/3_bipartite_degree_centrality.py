import networkx as nx
import nxviz as  nv
import matplotlib.pyplot as plt

G = nx.Graph()

numbers = range(3)

G.add_nodes_from(numbers, bipartite = 'customers')

letters = ['a', 'b']

G.add_nodes_from(letters, bipartite = 'products')

G.add_edges_from([(0,'a'),
                  ('b',2),
                  ('b',0)])
customer_nodes = [n for n, d in G.nodes(data=True) if d['bipartite']=="customers"]


print(nx.bipartite.degree_centrality(G, customer_nodes))