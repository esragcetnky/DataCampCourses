# Import matplotlib
import matplotlib.pyplot as plt
import networkx as nx  
import pandas as pd

fig = plt.figure()

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


# Create a list of the number of edges per month
edge_sizes = [len(g.edges()) for g in Gs]

# Plot edge sizes over time
plt.plot(edge_sizes)
plt.xlabel('Time elapsed from first month (in months).') 
plt.ylabel('Number of edges')                           
plt.show() 
