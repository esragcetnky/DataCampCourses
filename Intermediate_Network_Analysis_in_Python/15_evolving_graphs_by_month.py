import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from datetime import datetime

# --------------------------------------- PREPARE DATA -------------------------------------------------

# data = pd.read_csv('Data\college_message.txt',sep=" ",header=None,names=['source','target','date'])

# data['date'] = pd.to_datetime(data['date'],unit='s')
# data['year'] = data['date'].dt.year
# data['month'] = data['date'].dt.month
# data['day'] = data['date'].dt.day
# data['hour'] = data['date'].dt.hour
# data['minute'] = data['date'].dt.minute
# data['second'] = data['date'].dt.second

# print(data.head())

# data.to_csv('Data\college_message_df.csv', index=False)


data_processed = pd.read_csv('Data\college_message_df.csv')

print("------------------------- DATAFRAME VALUES -------------------------------------")
print("shape of dataset : ", data_processed.shape)
print("unique source number : ", len(data_processed['source'].unique()))
print("unique target number : ", len(data_processed['target'].unique()))
print("month unique : ", data_processed['month'].unique())

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
    
print("Total number of graphs group by month: ",len(Gs))

for x in range(len(Gs)):
    print(f"------------------------- GRAPH {x} VALUES -------------------------------------")
    print("Number of nodes :",len(Gs[x].nodes()))
    print("Number of edges :",len(Gs[x].edges()))