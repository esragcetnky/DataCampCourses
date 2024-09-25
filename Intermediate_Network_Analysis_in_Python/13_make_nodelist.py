import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


with open('Data\\american_revolution_graph.p', 'rb') as f:
    G = pickle.load(f)


# Initialize a list to store each edge as a record: nodelist
nodelist = []
for n, d in G.nodes(data=True):
    # nodeinfo stores one "record" of data as a dict
    nodeinfo = {'node': n} 
    
    # Update the nodeinfo dictionary 
    nodeinfo.update(d)
    
    # Append the nodeinfo to the node list
    nodelist.append(nodeinfo)
    

# Create a pandas DataFrame of the nodelist: node_df
node_df = pd.DataFrame(nodelist)
print(node_df.head())
