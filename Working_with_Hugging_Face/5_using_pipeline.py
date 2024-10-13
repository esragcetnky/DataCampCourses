from transformers import pipeline

input = "This course is pretty good, I guess."

input_turkish = "Harry Potter filmini sevmem." # means "I don't like Harry Potter movie"

# Create the task pipeline
task_pipeline = pipeline(task="sentiment-analysis")

# Create the model pipeline
model_pipeline = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")

# Predict the sentiment
task_output = task_pipeline(input)
model_output = model_pipeline(input)

# Predict the sentiment
task_output_2 = task_pipeline(input_turkish)
model_output_2 = model_pipeline(input_turkish)

print(f"Sentiment from task_pipeline: {task_output[0]['label']}; Sentiment from model_pipeline: {model_output[0]['label']}")

print(f"Sentiment from task_pipeline: {task_output_2[0]['label']}; Sentiment from model_pipeline: {model_output_2[0]['label']}")

