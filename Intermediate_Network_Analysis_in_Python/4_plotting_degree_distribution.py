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


# Get the 'users' nodes: user_nodes
user_nodes = get_nodes_from_partition(G, 'users')

# Compute the degree centralities: dcs
dcs = nx.degree_centrality(G)

# Get the degree centralities for user_nodes: user_dcs
user_dcs = [dcs[n] for n in user_nodes]

# Plot the degree distribution of users_dcs
plt.yscale('log')
plt.hist(user_dcs, bins=20)
plt.show()