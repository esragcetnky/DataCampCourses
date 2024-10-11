# Import the kmeans and vq functions
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.cluster.vq import kmeans, vq, whiten
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

fifa = pd.read_csv('Data/fifa_18_dataset.csv')

# Scale wage and value
fifa['scaled_sliding_tackle'] = whiten(fifa['sliding_tackle'])
fifa['scaled_aggression'] = whiten(fifa['aggression'])


# Generate cluster centers
cluster_centers, distortion = kmeans(fifa[['scaled_sliding_tackle', 'scaled_aggression']],
                                    2)


# Fit the data into a hierarchical clustering algorithm
distance_matrix = linkage(fifa[['scaled_sliding_tackle', 'scaled_aggression']], 'ward')

# Assign cluster labels
fifa['cluster_labels'], distortion_list = vq(fifa[['scaled_sliding_tackle', 'scaled_aggression']],cluster_centers)

# Plot clusters
sns.scatterplot(x='scaled_sliding_tackle', y='scaled_aggression', 
                hue='cluster_labels', data = fifa)
plt.show()

# Create a dendrogram
dn = dendrogram(distance_matrix)
# Display the dendogram
plt.show()