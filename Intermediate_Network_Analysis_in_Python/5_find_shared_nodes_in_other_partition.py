import networkx as nx
import nxviz as  nv
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import pandas as pd

github_data = pd.read_csv("Data\\github_edgelist.csv")

github_data['repo_id'] = 'r' + github_data['repo_id'].astype('str')
github_data['dev_id'] = 'u' + github_data['dev_id'].astype('str')
print("------------------------- DATAFRAME VALUES -------------------------------------")
print("shape of dataset : ", github_data.shape)
print("unique user number : ", len(github_data['dev_id'].unique()))
print("unique repo number : ", len(github_data['repo_id'].unique()))



H = nx.Graph()
H.add_nodes_from(github_data['dev_id'].unique(), bipartite='users')
H.add_nodes_from(github_data['repo_id'].unique(), bipartite='repos')

# for index, row in github_data.iterrows():
edges = [(row['dev_id'], row['repo_id']) for index, row in github_data.iterrows()]
H.add_edges_from(edges)
print("------------------------- GRAPH VALUES -------------------------------------")
print("Number of nodes :",len(H.nodes()))
print("Number of edges :",len(H.edges()))

with open('Data\\github_data_graph.p', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(H, f, pickle.HIGHEST_PROTOCOL)

with open('Data\\github_data_graph.p', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
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

user_list = get_nodes_from_partition(T, 'users')
top_10_users = sorted(nx.bipartite.degree_centrality(T, nodes=user_list).items(), key=lambda x:x[1], reverse=True)[:80]
print("-------------------------- TOP USERS ------------------------------------")
for x in top_10_users:
    if x[0] in user_list:
        print(x)

print("-------------------------- Shared Nodes between Users ------------------------------------")
print("Shared node between u71 and u367 : ",len(shared_partition_nodes(T, 'u71', 'u367')))
print("Shared node between u504 and u296 : ",len(shared_partition_nodes(T, 'u504', 'u296')))
print("Shared node between u504 and u62 : ",len(shared_partition_nodes(T, 'u504', 'u62')))
