# Import necessary modules
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx  
import pandas as pd

data_processed = pd.read_csv('Data\college_message_df.csv')
months = data_processed['month'].unique()

# Initialize an empty list: Gs
Gs = [] 
for month in months:
    # Instantiate a new undirected graph: G
    G = nx.Graph()
    
    # Add in all nodes that have ever shown up to the graph
    G.add_nodes_from(data_processed['source'])
    G.add_nodes_from(data_processed['target'])
    
    # Filter the DataFrame so that there's only the given month
    df_filtered = data_processed[data_processed['month'] == month]
    
    # Add edges from filtered DataFrame
    G.add_edges_from(zip(df_filtered['source'], df_filtered['target']))
    
    # Append G to the list of graphs
    Gs.append(G)

# Get the top 5 unique degree centrality scores: top_dcs
top_dcs = sorted(set(nx.degree_centrality(G).values()), reverse=True)[0:5]

# Create list of nodes that have the top 5 highest overall degree centralities
top_connected = []
for n, dc in nx.degree_centrality(G).items():
    if dc in top_dcs:
        top_connected.append(n)

# Create a defaultdict in which the keys are nodes and the values are a list of connectivity scores over time
connectivity = defaultdict(list)
for n in top_connected:
    for g in Gs:
        connectivity[n].append(len(list(g.neighbors(n))))

# Plot the connectivity for each node
fig = plt.figure() 
for n, conn in connectivity.items(): 
    plt.plot(conn, label=n) 
plt.legend()  
plt.show()