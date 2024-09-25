from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import yaml

# ----------------------------------------- Credential File  ---------------------------------------------------------#
CREDENTIALS_PATH = "credentials.yml"
credentials = yaml.safe_load(open(CREDENTIALS_PATH))

# Define an OpenAI chat model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=credentials['openai_api_key'])		

# Create a chat prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Respond to question: {question}")
    ]
)

# Chain the prompt template and model, and invoke the chain
llm_chain = prompt_template | llm
response = llm_chain.invoke({"question": "How can I retain learning?"})
print(response.content)