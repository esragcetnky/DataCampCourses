from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

image = Image.open("Data/florence_city.jpg")
image_2 = Image.open("Data/lake_side.jpg")


# Get the processor and model
processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

# Process the image
pixels = processor(images=image, return_tensors="pt").pixel_values

# Generate the ids
output = model.generate(pixel_values=pixels)

# Decode the output
caption = processor.batch_decode(output)



# Process the image
pixels_2 = processor(images=image_2, return_tensors="pt").pixel_values

# Generate the ids
output_2 = model.generate(pixel_values=pixels_2)

# Decode the output
caption_2 = processor.batch_decode(output_2)

print(f"Florence city : {caption[0]} \nLake side : {caption_2[0]} ")