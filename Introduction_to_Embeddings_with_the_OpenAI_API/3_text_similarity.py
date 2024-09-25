# % matplotlib inline
from openai import OpenAI
import yaml
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial import distance


# ----------------------------------------- Credential File  ---------------------------------------------------------#
CREDENTIALS_PATH = "credentials.yml"
credentials = yaml.safe_load(open(CREDENTIALS_PATH))

client = OpenAI(api_key = credentials['openai_api_key'])

# ----------------------------------------- Embedding headlines  ---------------------------------------------------------#
articles = [{"headline": "Economic Growth Continues Amid Global Uncertainty", "topic": "Business"},    
            {"headline": "Interest rates fall to historic lows", "topic": "Business"},    
            {"headline": "Scientists Make Breakthrough Discovery in Renewable Energy", "topic": "Science"},    
            {"headline": "India Successfully Lands Near Moon's South Pole", "topic": "Science"},    
            {"headline": "New Particle Discovered at CERN", "topic": "Science"},    
            {"headline": "Tech Company Launches Innovative Product to Improve Online Accessibility", "topic": "Tech"},    
            {"headline": "Tech Giant Buys 49% Stake In AI Startup", "topic": "Tech"},    
            {"headline": "New Social Media Platform Has Everyone Talking!", "topic": "Tech"},   
            {"headline": "The Blues get promoted on the final day of the season!", "topic": "Sport"},    
            {"headline": "1.5 Billion Tune-in to the World Cup Final", "topic": "Sport"}]

headline_text = [article["headline"] for article in articles]

print("----------------------------------- Headline Text array Starts Here -----------------------------------------------")
print(headline_text)
print("----------------------------------- Headline Text array Ends Here -----------------------------------------------")

# ----------------------------------------- Getting Embeddings for each headlines  ---------------------------------------------------------#
response = client.embeddings.create(model="text-embedding-3-small",
                                    input=headline_text)

response_dict = response.model_dump()

print("----------------------------------- Response Model dump Total Tokens Starts Here -----------------------------------------------")
print(response_dict['usage']["total_tokens"])
print("----------------------------------- Response Model dump Total Tokens Ends Here -----------------------------------------------")


# ----------------------------------------- Adding Embeddings to the array  ---------------------------------------------------------#
for i, article in enumerate(articles):
    article['embedding'] = response_dict['data'][i]['embedding']



search_word = "computer"
search_embeddings = client.embeddings.create(model="text-embedding-3-small",input=search_word)
search_embeddings = search_embeddings.data[0].embedding


distances = []

for article in articles:
    distance_value = distance.cosine( search_embeddings,article["embedding"])
    distances.append(distance_value)

index_of_smallest = np.argmin(distances)

print(articles[index_of_smallest]['headline'])



