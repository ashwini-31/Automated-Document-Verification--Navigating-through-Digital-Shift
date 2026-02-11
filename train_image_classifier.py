"""
Enhanced Training Script for Image-Based Document Classifier
Trains ResNet18 CNN on Tobacco3482 dataset with advanced techniques
"""

import os
import sys
import torch
from ml_modules.image_classifier import ImageDocumentClassifier
from collections import Counter

def main():
    """
    Enhanced main training function with advanced techniques
    """
    print("=" * 70)
    print("🚀 ENHANCED Image-Based Document Classifier Training")
    print("=" * 70)
    
    # Configuration
    dataset_path = 'datasets/Tobacco3482-jpg/'
    model_save_path = 'trained_models/image_doc_classifier.pth'
    num_epochs = 20
    learning_rate = 1e-4
    train_ratio = 0.8
    batch_size = 32
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please ensure Tobacco3482 dataset is available in datasets/ folder")
        print("Expected structure:")
        print("datasets/Tobacco3482-jpg/")
        print("├── ADVE/")
        print("├── Email/")
        print("├── Form/")
        print("├── Letter/")
        print("├── Memo/")
        print("├── News/")
        print("├── Note/")
        print("├── Report/")
        print("├── Resume/")
        print("└── Scientific/")
        return
    
    # Initialize classifier
    print("🔧 Initializing enhanced classifier...")
    classifier = ImageDocumentClassifier(model_save_path)
    
    try:
        # Load datasets with proper transforms
        print("📂 Loading training dataset...")
        train_dataset = classifier.load_dataset(dataset_path, transform_type='train')
        
        print("📂 Loading validation dataset...")
        val_dataset = classifier.load_dataset(dataset_path, transform_type='val')
        
        # Create data loaders
        print("🔄 Creating data loaders...")
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"✅ Dataset setup completed:")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        print(f"   Number of classes: {len(classifier.class_names)}")
        print(f"   Class names: {classifier.class_names}")
        
        # Display class distribution
        print("\n📊 Class Distribution Analysis:")
        train_targets = [label for _, label in train_dataset]
        val_targets = [label for _, label in val_dataset]
        
        train_counter = Counter(train_targets)
        val_counter = Counter(val_targets)
        
        for i, class_name in enumerate(classifier.class_names):
            train_count = train_counter.get(i, 0)
            val_count = val_counter.get(i, 0)
            total = train_count + val_count
            percentage = (total / (len(train_dataset) + len(val_dataset))) * 100
            print(f"   {class_name}: {total:4d} samples ({percentage:5.1f}%) - Train: {train_count:3d}, Val: {val_count:3d})")
        
        # Train enhanced model
        print("\n🚀 Starting enhanced training...")
        print(f"   Device: {classifier.device}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Batch size: {batch_size}")
        print(f"   Optimizer: AdamW with weight decay")
        print(f"   Scheduler: ReduceLROnPlateau")
        print(f"   Early stopping: Patience=5")
        print("-" * 70)
        
        training_history = classifier.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate
        )
        
        # Print final results
        print("\n" + "=" * 70)
        print("🎉 ENHANCED TRAINING COMPLETED!")
        print("=" * 70)
        
        final_train_acc = training_history['train_acc'][-1]
        final_val_acc = training_history['val_acc'][-1]
        best_val_acc = training_history['best_val_acc']
        best_epoch = training_history['best_epoch']
        
        print(f"📊 Final Results:")
        print(f"   Final Training Accuracy: {final_train_acc:.2f}%")
        print(f"   Final Validation Accuracy: {final_val_acc:.2f}%")
        print(f"   Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
        
        # Check if target achieved
        target_min = 75.0
        if best_val_acc >= target_min:
            print(f"🎯 TARGET ACHIEVED: {best_val_acc:.2f}% >= {target_min}%")
        else:
            print(f"⚠️ TARGET NOT ACHIEVED: {best_val_acc:.2f}% < {target_min}%")
        
        # Save model
        print("\n💾 Saving final model...")
        classifier.save_model()
        
        print(f"✅ Final model saved to: {model_save_path}")
        print(f"✅ Best model saved to: {model_save_path.replace('.pth', '_best.pth')}")
        
        # Test prediction
        print("\n🧪 Testing prediction...")
        if len(val_dataset) > 0:
            # Get a sample image from validation set
            sample_image, sample_label = val_dataset[0]
            
            # Convert tensor back to PIL if needed
            if isinstance(sample_image, torch.Tensor):
                from torchvision import transforms
                sample_image = transforms.ToPILImage()(sample_image)
            
            predicted_class, confidence = classifier.predict_from_pil_image(
                sample_image
            )
            actual_class = classifier.class_names[sample_label]
            
            print(f"   Sample prediction:")
            print(f"   Actual: {actual_class}")
            print(f"   Predicted: {predicted_class} (Confidence: {confidence:.3f})")
        
        # Generate confusion matrix
        print("\n📈 Generating confusion matrix...")
        generate_confusion_matrix(classifier, val_loader, classifier.class_names)
        
        print("\n✅ Enhanced training pipeline completed successfully!")
        print("You can now use the best model in the main application.")
        print(f"Use: trained_models/image_doc_classifier_best.pth")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def generate_confusion_matrix(classifier, val_loader, class_names):
    """
    Generate and display confusion matrix
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        classifier.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(classifier.device)
                outputs = classifier.model(images)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        print("\n📊 Confusion Matrix:")
        print("   Predicted →")
        print("   Actual ↓")
        
        # Print confusion matrix
        for i, true_class in enumerate(class_names):
            print(f"   {true_class:10s}", end="")
            for j, pred_class in enumerate(class_names):
                count = cm[i, j]
                print(f" {count:4d}", end="")
            print()
        
        # Calculate per-class accuracy
        print("\n📊 Per-Class Accuracy:")
        for i, class_name in enumerate(class_names):
            if cm[i, i] > 0:
                accuracy = cm[i, i] / cm[i, :].sum() * 100
                print(f"   {class_name:10s}: {accuracy:5.1f}%")
            else:
                print(f"   {class_name:10s}: N/A")
                
    except ImportError as e:
        print(f"⚠️ Cannot generate confusion matrix (missing dependencies): {e}")
        print("Install with: pip install matplotlib seaborn scikit-learn")

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
