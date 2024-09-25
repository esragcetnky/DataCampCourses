from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
import pandas as pd
import yaml

# ----------------------------------------- Credential File  ---------------------------------------------------------#
CREDENTIALS_PATH = "credentials.yml"
credentials = yaml.safe_load(open(CREDENTIALS_PATH))


llm = ChatOpenAI(model="gpt-4o-mini", api_key=credentials["openai_api_key"], temperature=0)

# Create a document loader for rag_vs_fine_tuning.pdf
loader = PyPDFLoader('Data\\Salesforce_Turkish_Report.pdf')

# Load the document
data = loader.load()

# Split the document using RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    separators = ["\n\n", "\n", " ", ""],
    chunk_size = 300,
    chunk_overlap = 50
)
docs = splitter.split_documents(data) 

embedding_function = OpenAIEmbeddings(api_key=credentials['openai_api_key'],
                                      model='text-embedding-3-small')

vectorstore = Chroma.from_documents(
    docs,
    embedding= embedding_function,
    persist_directory= ".\chromadb_rag_example",
)

retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":2}
)

# Add placeholders to the message string
message = """
Answer the following question using the context provided
Only use documents to answer the question, if the answer is not in the documents, write that you don't have the info:

Context:
{context}

Question:
{question}

Answer:
"""

# Create a chat prompt template from the message string
prompt_template = ChatPromptTemplate.from_messages([("human", message)])

# print(prompt_template)


# Create a chain to link retriever, prompt_template, and llm
rag_chain = ({"context": retriever, "question": RunnablePassthrough()}
            | prompt_template
            | llm)

while (True):
    string = str(input())
    # Invoke the chain
    response = rag_chain.invoke(string)
    print(response.content)