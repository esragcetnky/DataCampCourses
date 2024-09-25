import networkx as nx
import nxviz as  nv
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import pandas as pd


with open('Data\\github_data_graph.p', 'rb') as f:
    T = pickle.load(f)


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

def shared_partition_nodes(G, node1, node2):
    # Check that the nodes belong to the same partition
    assert G.nodes[node1]['bipartite'] == G.nodes[node2]['bipartite']

    # Get neighbors of node 1: nbrs1
    nbrs1 = G.neighbors(node1)
    # Get neighbors of node 2: nbrs2
    nbrs2 = G.neighbors(node2)

    # Compute the overlap using set intersections
    overlap = set(nbrs1).intersection(nbrs2)
    return overlap


def user_similarity(G, user1, user2, proj_nodes):
    # Check that the nodes belong to the 'users' partition
    assert G.nodes[user1]['bipartite'] == 'users'
    assert G.nodes[user2]['bipartite'] == 'users'

    # Get the set of nodes shared between the two users
    shared_nodes = shared_partition_nodes(G,user1,user2)

    # Return the fraction of nodes in the projects partition
    return len(shared_nodes) / len(proj_nodes)

# Compute the similarity score between users 'u4560' and 'u1880'
user_nodes = get_nodes_from_partition(T, 'users')
print("Shared node between u71 and u367 : ",user_similarity(T, 'u289', 'u367', user_nodes))
print("Shared node between u504 and u296 : ",user_similarity(T, 'u504', 'u296', user_nodes))
print("Shared node between u504 and u367 : ",user_similarity(T, 'u504', 'u367', user_nodes))
print("Shared node between u71 and u504 : ",user_similarity(T, 'u71', 'u504', user_nodes))

