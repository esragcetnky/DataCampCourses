import tiktoken
import chromadb

client = chromadb.PersistentClient("./chromadb_vectordb")



collection = client.get_collection(name = "first_collection")

enc = tiktoken.encoding_for_model("text-embedding-3-small")

print("# ----------------------------------------- Documents ---------------------------------------------------------#")
print(collection.get()["documents"])

total_tokens = sum(len(enc.encode(text)) for text in collection.get()["documents"]) 

cost_per_1k_tokens = 0.00002

print("# ----------------------------------------- Tokens & Cost ---------------------------------------------------------#")
print('total tokens :', total_tokens)

print('Cost :', cost_per_1k_tokens*total_tokens/1000)