import networkx as nx
import matplotlib.pyplot as plt
import nxviz as nv

G = nx.barbell_graph(m1=5, m2=1)

print(nx.betweenness_centrality(G))

a = nv.circos(G)
plt.show()


# Compute the betweenness centrality of T: bet_cen
bet_cen = nx.betweenness_centrality(G)

# Compute the degree centrality of T: deg_cen
deg_cen = nx.degree_centrality(G)

# Create a scatter plot of betweenness centrality and degree centrality
plt.scatter(list(bet_cen.values()), list(deg_cen.values()))

# Display the plot
plt.show()