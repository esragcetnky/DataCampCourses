from langchain_text_splitters import RecursiveCharacterTextSplitter

quote = '''One machine can do the work of fifty ordinary humans.\nNo machine can dothe work of one extraordinary human.'''

chunk_size = 24
chunk_overlap = 3

rc_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', " ", ""],
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap 
)

docs = rc_splitter.split_text(quote)

print("Docs : ",docs)
print("----------------------------------")
print([len(doc) for doc in docs])


print("----------------------------------")

quote = 'Words are flowing out like endless rain into a paper cup,\nthey slither while they pass,\nthey slip away across the universe.'
chunk_size = 24
chunk_overlap = 10

# Create an instance of the splitter class
splitter = RecursiveCharacterTextSplitter(
    separators =["\n"," ",""],
    chunk_overlap = chunk_overlap,
    chunk_size= chunk_size
)

# Split the document and print the chunks
docs = splitter.split_text(quote)
print(docs)
print("----------------------------------")
print([len(doc) for doc in docs])