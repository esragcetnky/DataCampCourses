from transformers import image_transforms, pipeline, AutoImageProcessor, ResNetForImageClassification
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch


original_image = Image.open("florence_city.jpg")
original_image_2 = Image.open("lake_side.jpg")

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = processor(original_image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()

print("Label for the image florence city : ",model.config.id2label[predicted_label])

inputs = processor(original_image_2, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()

print("Label for the image lake side  : ",model.config.id2label[predicted_label])


