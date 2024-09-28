# Import necessary modules
import matplotlib.pyplot as plt
from nxviz import arc
import pickle

# Create the un-customized Arc plot: a
T = pickle.load(open("Data\ego-twitter.p", 'rb'))
a = arc(T)
plt.show()

# Create the customized Arc plot: a2
a2 = arc(T,sort_by='category', node_color_by='category')

# Display the plot
plt.show()

print(T.nodes(data=True))