from transformers import pipeline
from datasets import Dataset
from datasets import load_dataset

dataset = Dataset.from_file("Data/common_language.arrow")
dataset_2 = load_dataset("yagmurx/ataturk_voice_restorated",split='train')
# # Save the old sampling rate
# Create the pipeline
classifier = pipeline(task="audio-classification", model="facebook/mms-lid-126")

# Extract the sample
audio = dataset[1]["audio"]["array"]
sentence = dataset[1]["sentence"]

# Predict the language
prediction = classifier(audio)


# Extract the sample
audio_2 = dataset_2[1]["audio"]["array"]

# Predict the language
prediction_2 = classifier(audio_2)


print("################################### Common Language #####################################################")
print(prediction)
print(f"Predicted language is '{prediction[0]['label'].upper()}' for the sentence '{sentence}'")




print("################################### Atatürk Voice #####################################################")
print(prediction_2)
print(f"Predicted language is '{prediction_2[0]['label'].upper()}''")