from huggingface_hub import HfApi, list_models
import yaml

# ----------------------------------------- Credential File  ---------------------------------------------------------#
CREDENTIALS_PATH = "credentials.yml"
credentials = yaml.safe_load(open(CREDENTIALS_PATH))

# Use root method
models = list_models()

api = HfApi(
    endpoint="https://huggingface.co", # Can be a Private Hub endpoint.
    token=credentials['huggingface_api_key'], # Token is not persisted on the machine.
)
# Return the filtered list from the Hub
models = api.list_models(
    filter="text-classification",
    sort="downloads",
    direction=-1,
  	limit=5
)

# Store as a list
modelList = list(models)

print([x.modelId for x in modelList])