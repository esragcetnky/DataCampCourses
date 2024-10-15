import librosa
from datasets import load_dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import DataLoader
import numpy as np

cv_13 = load_dataset("mozilla-foundation/common_voice_13_0", "hi", split="train")
batch_sampler = BatchSampler(RandomSampler(cv_13), batch_size=32, drop_last=False)
dataloader = DataLoader(cv_13, batch_sampler=batch_sampler)


# Create a list of durations
old_durations_list = []

# Loop over dataset
for row in dataset['path']:
    old_durations_list.append(librosa.get_duration(path=row))

# Creat a new column
dataset = dataset.add_column("duration", old_durations_list)

# Filter the dataset
filtered_dataset = dataset.filter(lambda d: d < 6.0, input_columns=["duration"], keep_in_memory=True)

# Save new durations
new_durations_list = filtered_dataset['duration']

print("Old duration:", np.mean(old_durations_list))
print("New duration:", np.mean(new_durations_list))