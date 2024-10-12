# Import the kmeans and vq functions
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.cluster.vq import kmeans, vq, whiten
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.image as img

# Read batman image and print dimensions
harry_potter_image = img.imread('harry_potter.jpg')
print(harry_potter_image.shape)

r = []
g = []
b = []

# Store RGB values of all pixels in lists r, g and b
for row in harry_potter_image:
    for temp_r, temp_g, temp_b in row:
        r.append(temp_r)
        g.append(temp_g)
        b.append(temp_b)

pixels = pd.DataFrame({'red': r,'blue': b,'green': g})

print(pixels.head())

# Scale wage and value
pixels['scaled_red'] = whiten(pixels['red'])
pixels['scaled_blue'] = whiten(pixels['blue'])
pixels['scaled_green'] = whiten(pixels['green'])

distortions = []
num_clusters = range(1, 11)

# Create a list of distortions from the kmeans function
for i in num_clusters :
    cluster_centers, distortion = kmeans(pixels[['scaled_red',
                                                    'scaled_blue',
                                                    'scaled_green']], i)
    distortions.append(distortion)

# Create a DataFrame with two lists, num_clusters and distortions
elbow_plot = pd.DataFrame({'num_clusters':num_clusters,'distortions':distortions})

# Create a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', data = elbow_plot)
plt.xticks(num_clusters)
plt.show()