import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


with open('Data\\american_revolution_graph.p', 'rb') as f:
    G = pickle.load(f)


# Initialize a list to store each edge as a record: edgelist
edgelist = []
for n1, n2, d in G.edges(data=True):
    # Initialize a dictionary that shows edge information: edgeinfo
    edgeinfo = {'node1':n1, 'node2':n2}
    
    # Update the edgeinfo data with the edge metadata
    edgeinfo.update(d)
    
    # Append the edgeinfo to the edgelist
    edgelist.append(edgeinfo)
    
# Create a pandas DataFrame of the edgelist: edge_df
edge_df = pd.DataFrame(edgelist)
print(edge_df.head())