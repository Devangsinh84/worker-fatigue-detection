import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = "data"
categories = {"open": [], "closed": []}

for img_path in Path(DATA_DIR).rglob("*.png"):
    parts = img_path.stem.split("_")
    if len(parts) >= 7:
        label = parts[6]
        if label == "1":
            categories["open"].append(str(img_path))
        elif label == "0":
            categories["closed"].append(str(img_path))

print(f"Open eyes:   {len(categories['open'])}")
print(f"Closed eyes: {len(categories['closed'])}")
print(f"Total:       {len(categories['open']) + len(categories['closed'])}")

sample_open   = cv2.imread(categories["open"][0], cv2.IMREAD_GRAYSCALE)
sample_closed = cv2.imread(categories["closed"][0], cv2.IMREAD_GRAYSCALE)

print(f"\nImage shape: {sample_open.shape}")
print(f"Pixel range: {sample_open.min()} - {sample_open.max()}")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Sample images from dataset", fontsize=14)

for i in range(5):
    img = cv2.imread(categories["open"][i], cv2.IMREAD_GRAYSCALE)
    axes[0, i].imshow(img, cmap="gray")
    axes[0, i].set_title("Open")
    axes[0, i].axis("off")

for i in range(5):
    img = cv2.imread(categories["closed"][i], cv2.IMREAD_GRAYSCALE)
    axes[1, i].imshow(img, cmap="gray")
    axes[1, i].set_title("Closed")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig("assets/sample_images.png")
plt.show()
print("\nSample image saved to assets/sample_images.png")