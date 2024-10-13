from transformers import pipeline 

# Create the pipeline
classifier = pipeline(task="text-classification", model="cross-encoder/qnli-electra-base")

# Predict the output
output = classifier("Where is the capital of France?, Brittany is known for their kouign-amann.")

print(output)

output2 = classifier("Where is the capital of France?, Paris")
print(output2)

output3 = classifier("Where is the capital of France?, Paris is the capital of France.")
print(output3)