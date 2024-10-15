from transformers import image_transforms, pipeline, AutoImageProcessor, ResNetForImageClassification
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Create the pipeline
dqa = pipeline(task="document-question-answering", model="impira/layoutlm-document-qa")

# Set the image and question
image = "invoice-template.png"
question = "What do you see in this picture ?"

# Get the answer
results = dqa(image=image, question=question)

print(results)