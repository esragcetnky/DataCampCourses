import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
from nltk.tokenize import word_tokenize
import re
from scipy.cluster.vq import kmeans, vq

movies_data = pd.read_csv('Data/movies_plot.csv')

print(movies_data.head())

def remove_noise(text, stop_words = []):    
    tokens = word_tokenize(text)    
    cleaned_tokens = []    
    for token in tokens:        
        token = re.sub('[^A-Za-z0-9]+', '', token)        
        if len(token) > 1 and token.lower() not in stop_words:            
            # Get lowercase            
            cleaned_tokens.append(token.lower())    
    return cleaned_tokens
        
        
print(remove_noise("It is lovely weather we are having.I hope the weather continues."))

# movies_data['noise_removed'] = ""

# for x, y in movies_data.iterrows():
#     movies_data['noise_removed'][x] = remove_noise(movies_data['Plot'][x])



from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=50,min_df=0.2, tokenizer=remove_noise)

tfidf_matrix = tfidf_vectorizer.fit_transform(movies_data['Plot'])

print(type(tfidf_matrix))

num_clusters = 2

# Generate cluster centers through the kmeans function
cluster_centers, distortion = kmeans(tfidf_matrix.todense(),num_clusters)

# Generate terms from the tfidf_vectorizer object
terms = tfidf_vectorizer.get_feature_names_out()

for i in range(num_clusters):
    # Sort the terms and print top 3 terms
    center_terms = dict(zip(terms, list(cluster_centers[i])))
    sorted_terms = sorted(center_terms, key=center_terms.get, reverse=True)
    print(sorted_terms[:3])