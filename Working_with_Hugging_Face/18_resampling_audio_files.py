from datasets import load_dataset, Audio
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import DataLoader

# # Save the old sampling rate
# old_sampling_rate = audio_file[1]["audio"]["sampling_rate"]

# # Resample the audio files
# audio_file = audio_file.cast_column("audio", Audio(sampling_rate=16_000))

# # Compare the old and new sampling rates
# print("Old sampling rate:", old_sampling_rate)
# print("New sampling rate:", audio_file[1]["audio"]["sampling_rate"])