# Import matplotlib
import matplotlib.pyplot as plt
import networkx as nx  
import pandas as pd
from scipy import stats


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


# Create a list of degree centrality scores month-by-month
cents = []
for G in Gs:
    cent = nx.degree_centrality(G)
    cents.append(cent)



fig = plt.figure()
for i in range(len(cents)):
    res = stats.ecdf(list(cents[i].values()))
    plt.plot(res.cdf.quantiles, res.cdf.probabilities, label='Month {0}'.format(i+1)) 

plt.legend()  
plt.xlabel('Degree centrality')
plt.ylabel('Cumulative score')
plt.show()
