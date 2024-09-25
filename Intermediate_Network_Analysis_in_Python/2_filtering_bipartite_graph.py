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

# Define get_nodes_from_partition()
def get_nodes_from_partition(G, partition):
    # Initialize an empty list for nodes to be returned
    nodes = []
    # Iterate over each node in the graph G
    for n in G.nodes():
        # Check that the node belongs to the particular partition
        if G.nodes[n]['bipartite'] == partition:
            # If so, append it to the list of nodes
            nodes.append(n)
    return nodes

# Print the number of nodes in the 'projects' partition
print('Number of product nodes :',len(get_nodes_from_partition(G, 'products')))

# Print the number of nodes in the 'users' partition
print('Number of customer nodes :',len(get_nodes_from_partition(G, 'customers')))