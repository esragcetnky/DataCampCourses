from sentence_transformers import SentenceTransformer

sentence = "Programmers, do you put your comments (before|after) the related code?"

# Create the first embedding model
embedder1 = SentenceTransformer("all-MiniLM-L6-v2")

# Embed the sentence
embedding1 = embedder1.encode([sentence])

# Create and use second embedding model
embedder2 = SentenceTransformer("sentence-transformers/paraphrase-albert-small-v2")
embedding2 = embedder2.encode([sentence])
 
# Compare the shapes
print(embedding1.shape == embedding2.shape)