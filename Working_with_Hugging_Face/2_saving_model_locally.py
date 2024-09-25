from transformers import AutoModel

modelId = "distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModel.from_pretrained(modelId)

model.save_pretrained(save_directory = f"models/{modelId}")