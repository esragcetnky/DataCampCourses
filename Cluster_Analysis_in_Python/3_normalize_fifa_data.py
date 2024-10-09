import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.cluster.vq import whiten

fifa = pd.read_csv('Data/fifa_18_sample_data.csv')
print(fifa.head())

# Scale wage and value
fifa['scaled_wage'] = whiten(fifa['eur_wage'])
fifa['scaled_value'] = whiten(fifa['eur_value'])


# Check mean and standard deviation of scaled values
print(fifa[['scaled_wage', 'scaled_value']].describe())

# Plot the two columns in a scatter plot
fifa.plot(x='scaled_wage', y='scaled_value', kind='scatter')
plt.show()
