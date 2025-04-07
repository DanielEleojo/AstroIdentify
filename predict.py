import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
import torch.nn.functional as F

from PIL import Image
import json


# Load the same model architecture
num_classes = 12 

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(model.fc.in_features, num_classes)
)

# Load saved weights
model.load_state_dict(torch.load("constellation_best_model.pt", map_location='cpu'))
model.eval()

with open("constellation_classes.json", "r") as f:
    class_names = json.load(f)


inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(model, image_path, transform, class_names):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)


        predicted_class = class_names[predicted.item()]
    return predicted_class, confidence.item()

image_path = "denoise_testing/thresholded_denoised_gemini_constellation.jpg"
prediction = predict_image(model, image_path, inference_transforms, class_names)
print("Predicted Constellation:", prediction)
