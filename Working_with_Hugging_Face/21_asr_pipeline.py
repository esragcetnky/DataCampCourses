from transformers import pipeline
from evaluate import load
from datasets import Dataset
import librosa

exp_array,sampling_rate = librosa.load("Data/capital_turkish.wav")

# Create an ASR pipeline using Meta's wav2vec model
meta_asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

# Predict the text from the example audio
meta_pred = meta_asr(exp_array)["text"].lower()

# Repeat for OpenAI's Whisper model
open_asr =pipeline("automatic-speech-recognition", model = "openai/whisper-tiny")
open_pred = open_asr(exp_array)["text"].lower()

print("################################################################")
# Print the prediction from both models
print("META:", meta_pred)
print("OPENAI:", open_pred)


# Create the word error rate metric
wer = load("wer")

# Save the true sentence of the example
true_sentence = "Türkiye'nin başkenti Ankara'dır."

# Compute the wer for each model prediction
meta_wer = wer.compute(predictions=[meta_pred], references=[true_sentence])
open_wer = wer.compute(predictions=[open_pred], references=[true_sentence])

print("################################################################")
print(f"The WER for the Meta model is {meta_wer} and for the OpenAI model is {open_wer}")