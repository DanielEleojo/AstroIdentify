import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# === Inference transform (for model input) ===
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess_night_sky(image_path):
    # Load in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Invert image so stars are white
    image = cv2.bitwise_not(image)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # Optional: adaptive threshold to isolate stars
    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=5
    )

    # Convert to 3-channel (model expects RGB input)
    final_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    final_pil = Image.fromarray(final_image)

    return final_pil
