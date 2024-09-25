from langchain_openai import OpenAI
import yaml

# ----------------------------------------- Credential File  ---------------------------------------------------------#
CREDENTIALS_PATH = "credentials.yml"
credentials = yaml.safe_load(open(CREDENTIALS_PATH))

# Define the LLM
llm = OpenAI(model="gpt-3.5-turbo-instruct", api_key=credentials['openai_api_key'])

# Predict the words following the text in question
question = 'Whatever you do, take care of your shoes'
output = llm.invoke(question)

print(output)