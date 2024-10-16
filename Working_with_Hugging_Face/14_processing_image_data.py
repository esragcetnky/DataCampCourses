from transformers import image_transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

original_image = Image.open("Data/florence_city.jpg")
# Create the numpy array
numpy_image = np.array(original_image)

print(numpy_image.shape)

# Crop the center of the image
cropped_image = image_transforms.center_crop(image=numpy_image, size=(200, 200))

imgplot = plt.imshow(cropped_image)
plt.show()