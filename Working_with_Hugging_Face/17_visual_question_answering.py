from transformers import pipeline

image = "lake_side.jpg"
question = "What do you see in this picture ?"

# Create pipeline
vqa = pipeline(task="visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")

# Use image and question in vqa
results = vqa(image=image, question=question)

print(results)