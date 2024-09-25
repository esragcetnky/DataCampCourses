# Import library
from langchain_community.document_loaders import PyPDFLoader

# Create a document loader for rag_vs_fine_tuning.pdf
loader = PyPDFLoader('Data\\Salesforce_Turkish_Report.pdf')

# Load the document
data = loader.load()
print(data[0])