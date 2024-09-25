import networkx as nx
import matplotlib.pyplot as plt
import nxviz as nv
import datetime
import pickle

T = pickle.load(open("Data\ego-twitter.p", 'rb'))
# Extract the nodes of interest: nodes
nodes = [n for n, d in T.nodes(data=True) if d['occupation'] == 'celebrity']

# Create the set of nodes: nodeset
nodeset = set(nodes)

# Iterate over nodes
for n in nodes:

    # Compute the neighbors of n: nbrs
    nbrs = T.neighbors(n)

    # Compute the union of nodeset and nbrs: nodeset
    nodeset = nodeset.union(nbrs)

# Compute the subgraph using nodeset: T_sub
T_sub = T.subgraph(nodeset)

# Draw T_sub to the screen
nx.draw(T_sub, with_labels=True)
plt.show()



# Compute the degree centralities of G: deg_cent
deg_cent = nx.degree_centrality(T)

# Compute the maximum degree centrality: max_dc
max_dc = max(deg_cent.values())

# Find the user(s) that have collaborated the most: prolific_collaborators
prolific_collaborators = [n for n, dc in deg_cent.items() if dc == max_dc]

# Print the most prolific collaborator(s)
print(prolific_collaborators)

# Compute the neighbors of n: nbrs
nbrs = T.neighbors(prolific_collaborators[0])


# Compute the subgraph using nodeset: T_sub
T_sub = T.subgraph(nbrs)

# Draw T_sub to the screen
nx.draw(T_sub, with_labels=True)
plt.show()