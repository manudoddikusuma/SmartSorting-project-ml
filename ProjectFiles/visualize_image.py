import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# Correct path: just point to 'fresh' folder
folder_path = r"C:\Users\kusuma\OneDrive\Desktop\SmartSorting\dataset_split\train\fresh"

# Get a random image name
img_name = random.choice(os.listdir(folder_path))
img_path = os.path.join(folder_path, img_name)

# Open and display the image
img = Image.open(img_path)
plt.imshow(img)
plt.title(img_name)
plt.axis('off')
plt.show()
