from openai import OpenAI
import yaml
from scipy.spatial import distance

# ----------------------------------------- Credential File  ---------------------------------------------------------#
CREDENTIALS_PATH = "credentials.yml"
credentials = yaml.safe_load(open(CREDENTIALS_PATH))


client = OpenAI(api_key=credentials['openai_api_key'])


# Define a create_embeddings function
def create_embeddings(texts):
  response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
  )
  response_dict = response.model_dump()
  
  return [data['embedding'] for data in response_dict['data']]


sentiments = [{'label': 'Positive'}, 
               {'label': 'Neutral'}, 
               {'label': 'Negative'}]

reviews = ['The food was delicious!',
 'The service was a bit slow but the food was good',
 'The food was cold, really disappointing!']



# Create a list of class descriptions from the sentiment labels
class_descriptions = [sentiment['label'] for sentiment in sentiments]


# Embed the class_descriptions and reviews
class_embeddings = create_embeddings(class_descriptions)
review_embeddings = create_embeddings(reviews)

def find_closest(query_vector, embeddings):
  distances = []
  for index, embedding in enumerate(embeddings):
    dist = distance.cosine(query_vector, embedding)
    distances.append({"distance": dist, "index": index})
  return min(distances, key=lambda x: x["distance"])

for index, review in enumerate(reviews):
  closest = find_closest(review_embeddings[index], class_embeddings)
  label = sentiments[closest['index']]['label']
  print(f'"{review}" was classified as {label}')