from langchain_community.document_loaders import UnstructuredHTMLLoader

# Create a document loader for unstructured HTML
loader = UnstructuredHTMLLoader("Data\\index.html")

# Load the document
data = loader.load()

# Print the first document
print(data[0])

# Print the first document's metadata
# print(data[0].metadata)