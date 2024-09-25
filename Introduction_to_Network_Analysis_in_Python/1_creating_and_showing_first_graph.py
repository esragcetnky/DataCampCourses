import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

G.add_nodes_from([1,2,3])
G.add_edge(1, 2)
G.nodes[1]['label'] = 'blue'

print(f"G.nodes : {G.nodes}")
print(f"G.edges : {G.edges}")
print(f"G.nodes(data=True) : {G.nodes(data=True)}")

nx.draw(G,with_labels=True)
plt.show()


