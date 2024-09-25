import networkx as nx
import pickle
import matplotlib.pyplot as plt

def nodes_with_m_nbrs(G, m):
    """
    Returns all nodes in graph G that have m neighbors.
    """
    nodes = set()

    # Iterate over all nodes in G
    for n in G.nodes():

        # Check if the number of neighbors of n matches m
        if len(list(G.neighbors(n))) == m:

            # Add the node n to the set
            nodes.add(n)

    # Return the nodes with m neighbors
    return nodes


# Create the matrix plot: m
T = pickle.load(open("Data\ego-twitter.p", 'rb'))
print('Neighbors of node 1: \n',list(T.neighbors(1)))


# Compute the degree of every node: degrees
degrees = [ len(list(T.neighbors(n))) for n in T.nodes()]

print("Degree of each node [:5]:")
# # Print the degrees
print(degrees[:5])

print("Degree centrality of each node [:5]:")
print(list(nx.degree_centrality(T).keys())[:5])
print(list(nx.degree_centrality(T).values())[:5])