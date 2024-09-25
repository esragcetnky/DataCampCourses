import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import csv
import yaml
from openai import OpenAI

# ----------------------------------------- Credential File  ---------------------------------------------------------#
CREDENTIALS_PATH = "credentials.yml"
credentials = yaml.safe_load(open(CREDENTIALS_PATH))

openai_client = OpenAI(api_key = credentials['openai_api_key'])

ids = []
documents = []

path = r"Data\netflix_titles.csv"

with open(path, encoding='utf-8') as csvfile:
  reader = csv.DictReader(csvfile)
  for i, row in enumerate(reader):
    ids.append(row['show_id'])        
    text = f"Title: {row['title']} ({row['type']})\nDescription: {row['description']}\nCategories: {row['listed_in']}"        
    documents.append(text)

embeddings = []

# Define a create_embeddings function
def create_embeddings(texts):
  response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
  )
  response_dict = response.model_dump()
  
  return [data['embedding'] for data in response_dict['data']]



client = chromadb.PersistentClient('./chromadb_vectordb')

collection = client.get_or_create_collection(
  name="netflix_titles",
  embedding_function=OpenAIEmbeddingFunction(model_name="text-embedding-3-small", 
                                             api_key=credentials['openai_api_key'])
)



# Add the documents and IDs to the collection
collection.add(
  ids=ids,
  documents=documents
)

# Query the collection for "films about dogs"
result = collection.query(
  query_texts=["films about dogs"],
  n_results = 3
)

print(result)