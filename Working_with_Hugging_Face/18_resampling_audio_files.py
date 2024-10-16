from datasets import Audio
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import DataLoader
import librosa
import matplotlib.pyplot as  plt
from datasets import Dataset

audio_file = Dataset.from_file("Data/common_language.arrow")

# # Save the old sampling rate
old_sampling_rate =  audio_file[1]["audio"]["sampling_rate"]

# # Resample the audio files
audio_file = audio_file.cast_column("audio", Audio(sampling_rate=16_000))

# Compare the old and new sampling rates
print("Old sampling rate:", old_sampling_rate)
print("New sampling rate:", audio_file[1]["audio"]["sampling_rate"])


audio_file_array = audio_file[1]["audio"]["array"]
sampling_rate = audio_file[1]["audio"]["sampling_rate"]

plt.figure().set_figwidth(12)
librosa.display.waveshow(audio_file_array, sr=sampling_rate)
plt.show()

