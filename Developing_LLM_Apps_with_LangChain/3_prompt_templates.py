from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
import yaml

# ----------------------------------------- Credential File  ---------------------------------------------------------#
CREDENTIALS_PATH = "credentials.yml"
credentials = yaml.safe_load(open(CREDENTIALS_PATH))

# Set your Hugging Face API token
huggingfacehub_api_token = credentials['huggingface_api_key']

# Create a prompt template from the template string
template = "You are an artificial intelligence assistant, answer the question. {question}"
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create a chain to integrate the prompt template and LLM
llm = HuggingFaceEndpoint(repo_id='tiiuae/falcon-7b-instruct', huggingfacehub_api_token=huggingfacehub_api_token, add_to_git_credential=True)
llm_chain = prompt | llm

question = "How does LangChain make LLM application development easier?"
print(llm_chain.invoke({"question": question}))