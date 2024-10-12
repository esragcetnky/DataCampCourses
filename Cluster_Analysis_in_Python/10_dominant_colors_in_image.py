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

colors = []

cluster_centers, distortion = kmeans(pixels[['red',
                                                    'blue',
                                                    'green']], 3)

# Get standard deviations of each color
r_std, g_std, b_std = pixels[['red', 'green', 'blue']].std()

for cluster_center in cluster_centers:
    scaled_r, scaled_g, scaled_b = cluster_center
    # Convert each standardized value to scaled value
    colors.append((
        scaled_r * r_std / 255,
        scaled_g * g_std / 255,
        scaled_b * b_std / 255
    ))

# Display colors of cluster centers
plt.imshow([colors])
plt.show()