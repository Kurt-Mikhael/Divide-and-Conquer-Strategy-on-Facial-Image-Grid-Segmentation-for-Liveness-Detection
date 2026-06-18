"""
Training script for Lightweight CNN Liveness Detection.

Trains a very small CNN (150K params) on CPU using the dataset.
Pure PyTorch without torchvision dependency.
"""
import os
import sys
import time
import json
from pathlib import Path
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import cv2

from liveness_detection.models.dl_cnn import (
    LightweightCNN, 
    get_model_summary
)


class LivenessDataset(data.Dataset):
    """
    Custom dataset for liveness detection.
    Loads images from 'real' and 'spoof' directories.
    """
    
    def __init__(self, dataset_dir: str, transform=None, grayscale: bool = False):
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.grayscale = grayscale
        self.samples = []
        self.class_names = []
        
        # Find class directories
        class_dirs = sorted([d for d in self.dataset_dir.iterdir() if d.is_dir()])
        
        for idx, class_dir in enumerate(class_dirs):
            self.class_names.append(class_dir.name)
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
            
            for img_path in image_files:
                self.samples.append((str(img_path), idx))
        
        print(f"Dataset loaded: {len(self.samples)} images")
        print(f"  Classes: {self.class_names}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image with OpenCV
        image = cv2.imread(img_path)
        if image is None:
            # Fallback to PIL
            image = np.array(Image.open(img_path).convert('RGB'))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if self.transform:
            image = self.transform(image, self.grayscale)
        
        return image, label


def default_transform(image: np.ndarray, grayscale: bool = False, input_size: int = 128):
    """
    Default transform: resize, convert, normalize.
    """
    if grayscale:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (input_size, input_size))
        image = image.astype(np.float32) / 255.0
        # Add channel dimension
        image = np.expand_dims(image, axis=0)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (input_size, input_size))
        image = image.astype(np.float32) / 255.0
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))
    
    return torch.from_numpy(image).float()


def augment_transform(image: np.ndarray, grayscale: bool = False, input_size: int = 128):
    """
    Transform with augmentation: random flip, rotation.
    """
    # Random horizontal flip
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)
    
    # Random rotation (-10 to 10 degrees)
    angle = np.random.uniform(-10, 10)
    if abs(angle) > 0.5:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    return default_transform(image, grayscale, input_size)


def train_model(dataset_dir: str, output_dir: str, 
                epochs: int = 5, batch_size: int = 32,
                lr: float = 0.001, input_size: int = 128,
                grayscale: bool = False, val_split: float = 0.2):
    """
    Train the lightweight CNN model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("LIGHTWEIGHT CNN TRAINING")
    print("="*60)
    
    # Check dataset
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return None
    
    # Check classes
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    print(f"Found classes: {[d.name for d in class_dirs]}")
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = LightweightCNN(num_classes=2, grayscale=grayscale)
    get_model_summary(model)
    
    # Load dataset
    print("\nLoading dataset...")
    full_dataset = LivenessDataset(str(dataset_path), transform=None, grayscale=grayscale)
    
    # Split dataset
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    torch.manual_seed(42)
    train_dataset, val_dataset = data.random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders with custom collate function
    def train_collate(batch):
        images, labels = zip(*batch)
        # Apply augmentation
        images = [augment_transform(img.numpy() if isinstance(img, torch.Tensor) else img, grayscale, input_size) for img in images]
        images = torch.stack(images)
        labels = torch.tensor(labels)
        return images, labels
    
    def val_collate(batch):
        images, labels = zip(*batch)
        images = [default_transform(img.numpy() if isinstance(img, torch.Tensor) else img, grayscale, input_size) for img in images]
        images = torch.stack(images)
        labels = torch.tensor(labels)
        return images, labels
    
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=0, collate_fn=train_collate)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=0, collate_fn=val_collate)
    
    class_names = full_dataset.class_names
    print(f"Dataset: {len(full_dataset)} total")
    print(f"  Train: {train_size}, Val: {val_size}")
    print(f"  Classes: {class_names}")
    
    # Training setup
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }
    
    best_val_acc = 0.0
    
    print(f"\nStarting training: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print("="*60)
    
    total_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        
        history["train_loss"].append(train_loss / len(train_loader))
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = output_dir / "best_model.pth"
            torch.save(model.state_dict(), best_path)
            print(f"  -> Best model saved! (val_acc={val_acc:.4f})")
    
    total_time = time.time() - total_start
    print(f"\nTraining completed in {total_time:.2f}s")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Save final model
    final_path = output_dir / "final_model.pth"
    torch.save(model.state_dict(), final_path)
    
    # Save history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot history
    plot_training_history(history, output_dir / "training_history.png")
    
    print(f"\nModel saved to: {output_dir}")
    print(f"  - Best model: {best_path}")
    print(f"  - Final model: {final_path}")
    
    return str(best_path)


def plot_training_history(history: dict, save_path: str):
    """Plot and save training history."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    axes[0].plot(epochs, history["train_loss"], 'b-', label='Train Loss')
    axes[0].plot(epochs, history["val_loss"], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history["train_acc"], 'b-', label='Train Acc')
    axes[1].plot(epochs, history["val_acc"], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].set_title('Training Accuracy')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to: {save_path}")


def main():
    # Default paths
    script_dir = Path(__file__).parent
    dataset_dir = script_dir.parent / "datasets"
    output_dir = script_dir.parent / "models" / "saved"
    
    # Check if dataset exists
    if not dataset_dir.exists():
        dataset_dir = Path("liveness_detection/datasets")
        output_dir = Path("liveness_detection/models/saved")
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    
    # Train model
    model_path = train_model(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=5,        # 5 epochs for quick training
        batch_size=32,
        lr=0.001,
        input_size=128,
        grayscale=False,
        val_split=0.2
    )
    
    if model_path:
        print(f"\n{'='*60}")
        print("TRAINING SUCCESSFUL")
        print(f"Model saved at: {model_path}")
        print("="*60)


if __name__ == "__main__":
    main()
