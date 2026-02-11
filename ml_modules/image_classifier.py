"""
Image-Based Document Classifier Module
Uses ResNet18 CNN trained on Tobacco3482 dataset for document type classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, random_split
import numpy as np
from PIL import Image
import os
from typing import Tuple, Dict, Any

class ImageDocumentClassifier:
    def __init__(self, model_path: str = 'trained_models/image_doc_classifier.pth'):
        """
        Initialize image-based document classifier
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = None
        self.transform = None
        self.val_transform = None
        self.is_trained = False
        
        # Enhanced training transforms with data augmentation
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Clean validation transforms (no augmentation)
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Default transform for inference (use validation transform)
        self.transform = self.val_transform
        
        # Default class names for Tobacco3482 dataset
        self.default_classes = [
            'ADVE', 'Email', 'Form', 'Letter', 'Memo', 
            'News', 'Note', 'Report', 'Resume', 'Scientific'
        ]
        
    def create_model(self, num_classes: int = 10) -> nn.Module:
        """
        Create ResNet18 model with custom final layer
        Enhanced with unfreezing last 2 residual blocks
        """
        # Load pretrained ResNet18
        model = models.resnet18(weights='IMAGENET1K_V1')
        
        # Freeze early layers, unfreeze last 2 residual blocks
        # ResNet18 has 4 residual blocks + final fc layer
        # We'll unfreeze layer3 and layer4 (last 2 blocks)
        for name, param in model.named_parameters():
            if "layer3" in name or "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Replace final layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
        return model
    
    def load_dataset(self, dataset_path: str, transform_type: str = 'train') -> datasets.ImageFolder:
        """
        Load dataset using ImageFolder with appropriate transforms
        """
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Use appropriate transforms
        if transform_type == 'train':
            transform = self.train_transform
        else:
            transform = self.val_transform
        
        dataset = datasets.ImageFolder(dataset_path, transform=transform)
        self.class_names = dataset.classes
        
        print(f"Dataset loaded successfully!")
        print(f"Number of classes: {len(dataset.classes)}")
        print(f"Class names: {dataset.classes}")
        print(f"Total samples: {len(dataset)}")
        
        return dataset
    
    def split_dataset(self, dataset, train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """
        Split dataset into train and validation sets with proper transforms
        """
        dataset_size = len(dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = dataset_size - train_size
        
        # Create train and validation datasets with proper transforms
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Apply correct transforms to each split
        train_dataset.dataset.transform = self.train_transform
        val_dataset.dataset.transform = self.val_transform
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        return train_loader, val_loader
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, 
                   num_epochs: int = 20, learning_rate: float = 1e-4) -> Dict[str, Any]:
        """
        Enhanced CNN model training with advanced techniques
        """
        # Create model
        self.model = self.create_model(num_classes=len(self.class_names))
        self.model.to(self.device)
        
        # Compute class weights for imbalanced dataset
        class_counts = {}
        for _, labels in train_loader:
            for label in labels:
                class_counts[label.item()] = class_counts.get(label.item(), 0) + 1
        
        total_samples = sum(class_counts.values())
        class_weights = {i: total_samples / (len(class_counts) * class_counts.get(i, 1)) 
                        for i in range(len(self.class_names))}
        class_weights_tensor = torch.tensor([class_weights[i] for i in range(len(self.class_names))], 
                                      dtype=torch.float32).to(self.device)
        
        # Define loss and optimizer with improvements
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_val_acc': 0.0,
            'best_epoch': 0
        }
        
        print(f"Enhanced training started on device: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Batch size: 32")
        print(f"Learning rate: {learning_rate}")
        print(f"Optimizer: AdamW with weight decay: 1e-4")
        print(f"Class weights: {class_weights}")
        print("-" * 60)
        
        # Early stopping variables
        patience_counter = 0
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                if batch_idx % 20 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                          f'Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            # Learning rate scheduler step
            scheduler.step(val_loss)
            
            # Store history
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)
            
            # Track best model
            if val_acc > training_history['best_val_acc']:
                training_history['best_val_acc'] = val_acc
                training_history['best_epoch'] = epoch + 1
                self.save_best_model()
                print(f"🏆 New best validation accuracy: {val_acc:.2f}% at epoch {epoch+1}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f'Epoch [{epoch+1}/{num_epochs}] completed:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Best Val Acc: {training_history["best_val_acc"]:.2f}% (Epoch {training_history["best_epoch"]})')
            print('-' * 60)
            
            # Early stopping
            if patience_counter >= 5:
                print(f"⏹️ Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
        self.is_trained = True
        return training_history
    
    def save_best_model(self):
        """
        Save only the best model
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        best_model_path = self.model_path.replace('.pth', '_best.pth')
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'model_architecture': 'resnet18_enhanced'
        }, best_model_path)
        
        print(f"Best model saved to {best_model_path}")
    
    def save_model(self):
        """
        Save the trained model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'model_architecture': 'resnet18'
        }, self.model_path)
        
        print(f"Model saved successfully to {self.model_path}")
    
    def load_model(self) -> bool:
        """
        Load the trained model
        """
        if not os.path.exists(self.model_path):
            print(f"No trained model found at {self.model_path}")
            return False
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.class_names = checkpoint['class_names']
            
            # Create model with correct number of classes
            self.model = self.create_model(num_classes=len(self.class_names))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self.is_trained = True
            print(f"Model loaded successfully from {self.model_path}")
            print(f"Classes: {self.class_names}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def predict_document_type(self, image_path: str) -> Tuple[str, float]:
        """
        Predict document type from image
        Returns (predicted_class, confidence_score)
        """
        if not self.is_trained:
            if not self.load_model():
                raise ValueError("Model not trained and no saved model found")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            
            predicted_class = self.class_names[predicted_idx.item()]
            confidence_score = confidence.item()
        
        return predicted_class, confidence_score
    
    def predict_from_pil_image(self, pil_image: Image.Image) -> Tuple[str, float]:
        """
        Predict document type from PIL Image object
        Returns (predicted_class, confidence_score)
        """
        if not self.is_trained:
            if not self.load_model():
                raise ValueError("Model not trained and no saved model found")
        
        # Preprocess image
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            
            predicted_class = self.class_names[predicted_idx.item()]
            confidence_score = confidence.item()
        
        return predicted_class, confidence_score
    
    def get_class_probabilities(self, image_path: str) -> Dict[str, float]:
        """
        Get probability distribution for all classes
        """
        if not self.is_trained:
            if not self.load_model():
                raise ValueError("Model not trained and no saved model found")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        return dict(zip(self.class_names, probabilities.cpu().numpy()))

# Global instance for reuse
_classifier_instance = None

def get_image_classifier():
    """
    Get or create image classifier instance
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ImageDocumentClassifier()
    return _classifier_instance

def predict_document_type(image_path: str) -> Tuple[str, float]:
    """
    Convenience function to predict document type from image
    """
    classifier = get_image_classifier()
    return classifier.predict_document_type(image_path)

def predict_from_pil_image(pil_image: Image.Image) -> Tuple[str, float]:
    """
    Convenience function to predict document type from PIL Image
    """
    classifier = get_image_classifier()
    return classifier.predict_from_pil_image(pil_image)

if __name__ == "__main__":
    # Test the classifier
    classifier = ImageDocumentClassifier()
    print("Image Document Classifier initialized successfully!")
    print(f"Device: {classifier.device}")
    print(f"Default classes: {classifier.default_classes}")
