# Import modules
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
import pandas as pd
from transformers import pipeline, Trainer, TrainingArguments
from evaluate import load
from datasets import Dataset, load_dataset

data = load_dataset( "imdb")

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer_id_or_path = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer_max_len = 512
tokenizer_config = {'pretrained_model_name_or_path': tokenizer_id_or_path,
                            'max_len': tokenizer_max_len}
tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Use tokenizer on text
data = data.map(lambda row: tokenizer(row["text"], 
                                      return_tensors='pt', 
                                      padding='max_length', 
                                      truncation=True, 
                                      max_length=tokenizer.model_max_length), keep_in_memory=True)

# Create training arguments
training_args = TrainingArguments(output_dir="./results")

# Create the trainer
trainer = Trainer(
    model=model, 
    args=training_args, 
    data_collator= data_collator,
    train_dataset=data['train'], 
    eval_dataset=data['test']
)

# Start the trainer
trainer.train()

local_path = "./fine_tuned_model"

trainer.save_model(local_path)

text_example = "I am a HUGE fan of romantic comedies."

# Create the classifier
classifier = pipeline(task="sentiment-analysis", model="./fine_tuned_model")

# Classify the text
results = classifier(text=text_example)


print("#####################################################")
print(results)
print("#####################################################")
