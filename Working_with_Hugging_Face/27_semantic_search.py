from sentence_transformers import SentenceTransformer, util

# Create the first embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

sentences = ['Programmers, do you put your comments (before|after) the related code?',
 'How sure are we that there were never any intelligent dinosaurs?',
 'Can anyone suggest a desktop book reader for Mac that works similar to Stanza on the iPhone?',
 'I will be in Lima, Ohio Monday night/tuesday on business. What is there to do, and see in the area?',
 "I'm looking for a good quality headset that doesn't cost too much. Any recommendations?",
 'How do I get a list of all the duplicate items using LINQ?',
 "Please help me figure out why it's so tough for me to connect to Valve games. It's driving me insane.",
 "Is there such a thing as 'good' instant coffee?",
 'How do I get the distinct/unique values in a column in Excel?']


query = "I need a desktop book reader for Mac"

# Generate embeddings
query_embedding = embedder.encode([query])[0]
sentence_embeddings = embedder.encode([sentences])[0]

# Compare embeddings
hits = util.semantic_search(query_embedding, sentence_embeddings, top_k=2)

# Print the top results
for hit in hits[0]:
    print(sentences[hit["corpus_id"]], "(Score: {:.4f})".format(hit["score"]))