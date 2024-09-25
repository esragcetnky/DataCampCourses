from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Create a document loader for unstructured HTML
loader = UnstructuredHTMLLoader("Data\\index.html")


data = loader.load()

# Define variables
chunk_size = 300
chunk_overlap = 100

# Split the HTML
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=".")

docs = splitter.split_documents(data)
print(docs[:1])