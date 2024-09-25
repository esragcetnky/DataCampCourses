from openai import OpenAI
import yaml

# ----------------------------------------- Credential File  ---------------------------------------------------------#
CREDENTIALS_PATH = "credentials.yml"
credentials = yaml.safe_load(open(CREDENTIALS_PATH))

client = OpenAI(api_key = credentials['openai_api_key'])

response = client.embeddings.create(
    model = "text-embedding-3-small",
    input = "Embeddings are a numerical representation of text that can be used tomeasure the relatedness between two pieces of text."
)

response_dict = response.model_dump()

print(response_dict)