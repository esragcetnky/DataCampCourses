from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# This time I wanna try something neutral in 2 language to see what model will respond.
input_english = "Capital of Turkey is Ankara."
input_turkish = "Ankara Türkiye'nin başkentidir."

# Download the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Create the pipeline
sentimentAnalysis = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)

# Predict the sentiment
output_english = sentimentAnalysis(input_english)

# Predict the sentiment
output_turkish = sentimentAnalysis(input_english)

print(f"Sentiment using AutoClasses English: {output_english[0]['label']}")
print(f"Sentiment using AutoClasses Turkish: {output_turkish[0]['label']}")