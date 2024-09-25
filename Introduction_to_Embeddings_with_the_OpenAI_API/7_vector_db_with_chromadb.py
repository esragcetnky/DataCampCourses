import yaml
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# ----------------------------------------- Credential File  ---------------------------------------------------------#
CREDENTIALS_PATH = "credentials.yml"
credentials = yaml.safe_load(open(CREDENTIALS_PATH))

client = chromadb.PersistentClient("./chromadb_vectordb")


collection = client.get_or_create_collection(
    name = "first_collection",
    embedding_function= OpenAIEmbeddingFunction(
        model_name = "text-embedding-3-small",
        api_key = credentials['openai_api_key']
    )
)

# ----------------------------------------- List Collections  ---------------------------------------------------------#
print(f"List collections : {client.list_collections()}")

# ----------------------------------------- Inserting Embeddings to Collections  ---------------------------------------------------------#
collection.add(ids=["my-doc"],
               documents=["This is the source text."])

collection.add(ids=["my-doc-1", "my-doc-2"],
               documents=["This is the source text 1", "This is the source text 2"])



# ----------------------------------------- Inspectiong Collections  ---------------------------------------------------------#
print(f"Counting documents in a collection : {collection.count()}")

print(f"Peeking at the first 10 items : {collection.peek()}")

print(f"Retrieving items: {collection.get(ids=['my-doc-1'])}")