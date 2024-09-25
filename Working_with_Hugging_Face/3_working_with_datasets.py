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

# configuration parameter
# data = load_dataset("wikipedia' )

