import networkx as nx
import nxviz as nv
import pickle
import matplotlib.pyplot as plt


dg = pickle.load(open("Data\ego-twitter.p", 'rb'))

# nv.arc(dg)

# Create the circos plot: c
nv.circos(dg)

plt.show()