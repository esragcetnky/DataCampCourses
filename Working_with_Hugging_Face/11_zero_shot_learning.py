from transformers import pipeline

text = 'A 75-million-year-old Gorgosaurus fossil is the first tyrannosaur skeleton ever found with a filled stomach.'

# Build the zero-shot classifier
classifier = pipeline(task="zero-shot-classification", model="facebook/bart-large-mnli")

# Create the list
candidate_labels = ["politics", "science", "sports"]

# Predict the output
output = classifier(text, candidate_labels)

print(f"Top Label: {output['labels'][0]} with score: {output['scores'][0]}")