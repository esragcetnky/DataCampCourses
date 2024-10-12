import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.cluster.vq import whiten
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.cluster.vq import kmeans, vq

fifa = pd.read_csv('Data/fifa_18_sample_data.csv')

features_names = ['pac', 'sho', 'pas', 'dri', 'def', 'phy']
scaled_features = []

for x in features_names:
    # Scale wage and value
    fifa['scaled_'+x] = whiten(fifa[x])
    scaled_features.append('scaled_'+x)


# Create centroids with kmeans for 2 clusters
cluster_centers,_ = kmeans(fifa[scaled_features], 2)

# Assign cluster labels
fifa['cluster_labels'], distortion_list = vq(fifa[scaled_features],cluster_centers)

print(fifa.groupby('cluster_labels')[scaled_features].mean())


# Plot cluster centers to visualize clusters
fifa.groupby('cluster_labels')[scaled_features].mean().plot(legend=True, kind='bar')
plt.show()

# Get the name column of first 5 players in each cluster
for cluster in fifa['cluster_labels'].unique():
    print(cluster, fifa[fifa['cluster_labels'] == cluster]['name'].values[:5])