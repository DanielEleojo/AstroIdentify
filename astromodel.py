import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


def train_model():
    # Image transformations
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),    # Resize to match ResNet’s expected size
        #transforms.RandomRotation(10),   #Experimental
        #transforms.RandomHorizontalFlip(), # Light augmentation Experimental
        #transforms.ColorJitter(brightness=0.1, contrast=0.1), #Experimental       
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet means
            std=[0.229, 0.224, 0.225]     # ImageNet stds
        )
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Path to your dataset
    data_dir = r"C:\Users\DBABA\OneDrive\Documents\Data Science Projects\AstroIdentify\Constalations-Classification-1"


    # Create datasets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    val_dataset   = datasets.ImageFolder(os.path.join(data_dir, 'valid'),   transform=val_transforms)
    test_dataset  = datasets.ImageFolder(os.path.join(data_dir, 'test'),  transform=val_transforms)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=2)

    # Number of classes
    num_classes = len(train_dataset.classes)
    #print("Classes:", train_dataset.classes)

    # Load pretrained ResNet
    model = models.resnet18(pretrained=True)

    # Freeze all layers if you only want to train the final layer:
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final classification layer (fully connected)
    # ResNet18’s last layer is `model.fc`, which outputs 1000 classes by default (for ImageNet)
    
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    '''
    Experimental
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),                      # ← Add dropout before FC layer
        nn.Linear(num_features, num_classes)
    )
    '''
       

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Since we replaced only the final layer, we only pass the final layer’s parameters to the optimizer.
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)  #Experimental

    # If you decide to unfreeze more layers, pass those parameters as well:
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
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
        
        scheduler.step()   #Experimental
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                v_loss = criterion(val_outputs, val_labels)
                
                val_loss += v_loss.item() * val_images.size(0)
                _, val_predicted = torch.max(val_outputs, 1)
                val_correct += (val_predicted == val_labels).sum().item()
                val_total += val_labels.size(0)
        
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    test_acc = 100.0 * test_correct / test_total
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    return model, val_transforms, train_dataset.classes

from PIL import Image

def predict_image(model, image_path, transform, class_names, device):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        _, pred_idx = torch.max(output, 1)
    predicted_class = class_names[pred_idx.item()]
    return predicted_class

if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Step 1: Train the model and get what you need
    model, val_transforms, class_names = train_model()

    # Step 2: Predict on new image
    test_image_path = r'C:\Users\DBABA\OneDrive\Documents\Data Science Projects\AstroIdentify\denoise_testing\Aquarius_input.png'
    prediction = predict_image(model, test_image_path, val_transforms, class_names, device)
    print("Predicted Constellation:", prediction)
    
