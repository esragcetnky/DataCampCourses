from datasets import load_dataset_builder
from datasets import load_dataset

data_builder = load_dataset_builder("rotten_tomatoes")

print("------------------------ Description --------------------------")
print(data_builder.info.description)
print("------------------------ Features Info --------------------------")
print(data_builder.info.features)


data = load_dataset( "imdb")

# split_parameter
data = load_dataset("imdb", split='train')

print(data)

# Filter the documents
filtered = data.filter(lambda row: "football" in row["text"])

# Create a sample dataset
example = filtered.select(range(1))

print(example[0]["text"])