import networkx as nx
import nxviz as  nv
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import pandas as pd


with open('Data\\github_data_graph.p', 'rb') as f:
    T = pickle.load(f)


def recommend_repositories(G, from_user, to_user):
    # Get the set of repositories that from_user has contributed to
    from_repos = set(G.neighbors(from_user))
    # Get the set of repositories that to_user has contributed to
    to_repos = set(G.neighbors(to_user))

    # Identify repositories that the from_user is connected to that the to_user is not connected to
    return from_repos.difference(to_repos)

print("------------------------------------------------------------")
# Print the repositories to be recommended
print("Recommending from 'u7909' to 'u2148' :",recommend_repositories(T, 'u367', 'u71'))