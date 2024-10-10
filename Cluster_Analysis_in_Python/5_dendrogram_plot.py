import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.cluster.vq import whiten
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

x_coordinates = [80.1, 93.1, 86.6, 98.5, 86.4, 9.5, 15.2, 3.4, 10.4, 20.3, 44.2, 56.8, 49.2, 62.5, 44.0]
y_coordinates = [87.2, 96.1, 95.6, 92.4, 92.4, 57.7, 49.4, 47.3, 59.1, 55.5, 25.6, 2.1, 10.9, 24.1, 10.3]

df = pd.DataFrame({'x_coordinate': x_coordinates,
                   'y_coordinate': y_coordinates,
                   'x_scaled':whiten(x_coordinates),
                   'y_scaled':whiten(y_coordinates)})

# Use the linkage() function
distance_matrix = linkage(df[['x_scaled', 'y_scaled']], method = 'ward', metric = 'euclidean')

# Create a dendrogram
dn = dendrogram(distance_matrix)

# Display the dendogram
plt.show()