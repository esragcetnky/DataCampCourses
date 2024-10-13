from transformers import pipeline
from datasets import load_dataset_builder
from datasets import load_dataset

# configuration parameter
wiki = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train",)


# Create the list
text_to_summarize = [w["text"] for w in wiki]

# Create the pipeline
summarizer = pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum", min_length=20, max_length=50)

# Summarize each item in the list
summaries = summarizer(text_to_summarize[:3], truncation=True)

# Create for-loop to print each summary
for i in range(0,3):
  print(f"Summary {i+1}: {summaries[i]['summary_text']}")