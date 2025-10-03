"""
Constellation Classifier - Web-Ready Module
A clean, modular class for constellation classification that can be easily integrated into web applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
import json
import os
import numpy as np
from typing import Tuple, List, Dict, Optional


class ConstellationClassifier:
    """
    A web-ready constellation classifier using ResNet18.
    
    This class encapsulates all the functionality needed for constellation prediction
    in a clean interface suitable for web applications.
    """
    
    def __init__(self, model_path: str = "constellation_best_model.pt", 
                 classes_path: str = "constellation_classes.json"):
        """
        Initialize the constellation classifier.
        
        Args:
            model_path: Path to the trained model weights
            classes_path: Path to the JSON file containing class names
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.classes_path = classes_path
        self.model = None
        self.class_names = None
        self.transform = None
        
        # Initialize the classifier
        self._load_class_names()
        self._create_model()
        self._load_model_weights()
        self._setup_transforms()
        
    def _load_class_names(self):
        """Load constellation class names from JSON file."""
        try:
            with open(self.classes_path, 'r') as f:
                self.class_names = json.load(f)
            print(f"Loaded {len(self.class_names)} constellation classes")
        except FileNotFoundError:
            raise FileNotFoundError(f"Class names file not found: {self.classes_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {self.classes_path}")
    
    def _create_model(self):
        """Create the ResNet18 model architecture."""
        num_classes = len(self.class_names)
        
        # Load pretrained ResNet18
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Replace the final classification layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, num_classes)
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        print(f"Model created and moved to {self.device}")
    
    def _load_model_weights(self):
        """Load the trained model weights."""
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"Model weights loaded from {self.model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model weights file not found: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model weights: {str(e)}")
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms."""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet means
                               [0.229, 0.224, 0.225])   # ImageNet stds
        ])
    
    def _preprocess_image(self, image) -> torch.Tensor:
        """
        Preprocess an image for model inference.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL Image or numpy array")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms and add batch dimension
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def predict(self, image, return_probabilities: bool = False, 
                top_k: int = 3) -> Dict:
        """
        Predict constellation from an image.
        
        Args:
            image: PIL Image, numpy array, or path to image file
            return_probabilities: Whether to return class probabilities
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary containing prediction results
        """
        # Handle different input types
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            image = Image.open(image)
        
        # Preprocess image
        image_tensor = self._preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            # Prepare results
            predictions = []
            for i in range(top_k):
                pred_idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                predictions.append({
                    'constellation': self.class_names[pred_idx],
                    'confidence': prob,
                    'confidence_percent': f"{prob:.1%}"
                })
            
            result = {
                'predicted_constellation': predictions[0]['constellation'],
                'confidence': predictions[0]['confidence'],
                'confidence_percent': predictions[0]['confidence_percent'],
                'top_predictions': predictions
            }
            
            if return_probabilities:
                result['all_probabilities'] = {
                    self.class_names[i]: probabilities[0][i].item() 
                    for i in range(len(self.class_names))
                }
        
        return result
    
    def predict_batch(self, images: List, batch_size: int = 32) -> List[Dict]:
        """
        Predict constellations for a batch of images.
        
        Args:
            images: List of images (PIL Images, numpy arrays, or file paths)
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = [self.predict(img) for img in batch]
            results.extend(batch_results)
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_architecture': 'ResNet18',
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'device': str(self.device),
            'model_path': self.model_path,
            'input_size': (224, 224)
        }
    
    def save_prediction_to_file(self, image_path: str, output_path: str = None):
        """
        Predict and save results to a JSON file.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the results (optional)
        """
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{base_name}_prediction.json"
        
        result = self.predict(image_path, return_probabilities=True)
        result['input_image'] = image_path
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Prediction saved to {output_path}")


def create_classifier(model_path: str = None, classes_path: str = None) -> ConstellationClassifier:
    """
    Factory function to create a constellation classifier.
    
    Args:
        model_path: Path to model weights (uses default if None)
        classes_path: Path to class names file (uses default if None)
        
    Returns:
        Initialized ConstellationClassifier instance
    """
    kwargs = {}
    if model_path:
        kwargs['model_path'] = model_path
    if classes_path:
        kwargs['classes_path'] = classes_path
    
    return ConstellationClassifier(**kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Initialize classifier
    classifier = ConstellationClassifier()
    
    # Print model info
    info = classifier.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with a sample image if available
    test_image_path = "denoise_testing/thresholded_denoised_aquarius_constellation_2.jpg"
    if os.path.exists(test_image_path):
        print(f"\nTesting with {test_image_path}:")
        result = classifier.predict(test_image_path)
        print(f"Predicted: {result['predicted_constellation']} ({result['confidence_percent']})")
        print("\nTop 3 predictions:")
        for i, pred in enumerate(result['top_predictions'], 1):
            print(f"  {i}. {pred['constellation']}: {pred['confidence_percent']}")
    else:
        print(f"\nTest image not found: {test_image_path}")