# Import networkx
import networkx as nx
import pandas as pd
import pickle

# Read in the data: g
data = pd.read_csv('Data\\american-revolution.csv')


club_name = ['StAndrewsLodge','LoyalNine','NorthCaucus','LongRoomClub','TeaParty','BostonCommittee','LondonEnemies']
data = data.rename(columns={'Unnamed: 0':'Name'})

print(data.head())
G = nx.Graph()
G.add_nodes_from(data['Name'].unique(), bipartite='people')
G.add_nodes_from(club_name, bipartite='clubs')

for index, row in data.iterrows():
    for column in club_name:
        if row[column]==1:
            G.add_edges_from([(row['Name'], column)])

print("------------------------- DATAFRAME VALUES -------------------------------------")
print("shape of dataset : ", data.shape)
print("unique user number : ", len(data['Name'].unique()))
print("unique repo number : ", len(club_name))
print("------------------------- GRAPH VALUES -------------------------------------")
print("Number of nodes :",len(G.nodes()))
print("Number of edges :",len(G.edges()))

with open('Data\\american_revolution_graph.p', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

# Prepare the nodelists needed for computing projections: people, clubs
# This exercise shows you two ways to do it, one with `data=True` and one without.
people = [n for n in G.nodes() if G.nodes[n]['bipartite'] == 'people']
clubs = [n for n, d in G.nodes(data=True) if d['bipartite'] == 'clubs']

# Compute the people and clubs projections: peopleG, clubsG
peopleG = nx.bipartite.projected_graph(G, people)
clubsG = nx.bipartite.projected_graph(G, clubs)

print("-------------------- People Graph -----------------------")
print(peopleG)
print("-------------------- Clubs Graph -----------------------")
print(clubsG)