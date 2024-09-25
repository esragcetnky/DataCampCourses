# Import library
from langchain_community.document_loaders.csv_loader import CSVLoader

# Create a document loader for fifa_countries_audience.csv
loader = CSVLoader("Data\\customers_1.csv")

# Load the document
data = loader.load()
print(data[0])