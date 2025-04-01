import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from PIL import Image

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Basic, safe transforms (no augmentation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # === UPDATE THIS PATH ===
    data_path = r"C:\Users\DBABA\OneDrive\Documents\Data Science Projects\AstroIdentify\Constalations-Classification-1\train"

    # Load full dataset from training folder
    full_dataset = datasets.ImageFolder(root=data_path, transform=transform)

    # Check class labels and count
    print("Classes:", full_dataset.classes)
    print("Total training images:", len(full_dataset))

    # Split off 20 samples to test overfitting
    tiny_dataset, _ = random_split(full_dataset, [20, len(full_dataset) - 20])
    tiny_loader = DataLoader(tiny_dataset, batch_size=4, shuffle=True)

    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = True  # fine-tune all layers

    # Replace final layer with number of constellation classes
    num_classes = len(full_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train for 10 epochs on tiny dataset
    print("Starting training on 20 images...")
    for epoch in range(1, 11):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        
        for images, labels in tiny_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = 100.0 * correct / total
        avg_loss = running_loss / total
        print(f"Epoch [{epoch}/10] - Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()
