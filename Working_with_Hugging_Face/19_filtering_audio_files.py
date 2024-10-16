import librosa
import numpy as np
from datasets import load_dataset

dataset = load_dataset("yagmurx/ataturk_voice_restorated",split='train')

print("Dataset shape :",dataset.shape)
print(dataset['audio'][1]['path'])
# Create a list of durations
old_durations_list = []

# Loop over dataset
for row in range(len(dataset['audio'])):
    old_durations_list.append(librosa.get_duration(y=dataset[row]['audio']['array'], sr=dataset[row]['audio']['sampling_rate']))

# Creat a new column
dataset = dataset.add_column("duration", old_durations_list)

print(dataset['duration'])
# Filter the dataset
filtered_dataset = dataset.filter(lambda d: d < 50, input_columns=["duration"], keep_in_memory=True)

# Save new durations
new_durations_list = filtered_dataset['duration']

print("Old duration:", np.mean(old_durations_list))
print("New duration:", np.mean(new_durations_list))