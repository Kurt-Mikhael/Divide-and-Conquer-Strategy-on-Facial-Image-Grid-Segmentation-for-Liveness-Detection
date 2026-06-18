"""
Lightweight Deep Learning CNN for Liveness Detection.

This model uses a very small custom CNN architecture (less than 500K params)
that can run efficiently on CPU for comparison with the Divide and Conquer algorithm.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List
from PIL import Image
import json


class LightweightCNN(nn.Module):
    """
    Lightweight CNN for binary classification (real vs spoof).
    
    Architecture:
    - Input: 128x128x3 (RGB) or 1x128x128 (Grayscale)
    - Conv layers: 16 -> 32 -> 64 channels
    - FC layers: 64*16*16 -> 128 -> 2
    - Total params: ~150K (very small for CPU)
    """
    
    def __init__(self, num_classes: int = 2, grayscale: bool = False):
        super(LightweightCNN, self).__init__()
        self.grayscale = grayscale
        in_channels = 1 if grayscale else 3
        
        # Feature extraction
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # After 3 maxpool layers: 128 -> 64 -> 32 -> 16
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def get_name(self):
        return f"LightweightCNN({'gray' if self.grayscale else 'rgb'})"


class SimpleCNNDetector:
    """
    Wrapper class that provides the same interface as other detectors.
    
    This wraps the PyTorch CNN model so it can be used in the same pipeline.
    """
    
    def __init__(self, model_path: str = None, grayscale: bool = False, 
                 input_size: int = 128, device: str = None):
        """
        Initialize the CNN detector.
        
        Args:
            model_path: Path to saved model weights
            grayscale: Whether to use grayscale input
            input_size: Input image size (square)
            device: 'cuda' or 'cpu' (auto if None)
        """
        self.grayscale = grayscale
        self.input_size = input_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = LightweightCNN(num_classes=2, grayscale=grayscale)
        self.model.to(self.device)
        self.model.eval()
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def load(self, model_path: str):
        """Load model weights from file."""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def save(self, model_path: str):
        """Save model weights to file."""
        torch.save(self.model.state_dict(), model_path)
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for CNN.
        
        Args:
            image: BGR image (numpy array)
        
        Returns:
            torch.Tensor: Preprocessed image tensor (float32)
        """
        if self.grayscale:
            # Convert to grayscale
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (self.input_size, self.input_size))
            image = image.astype(np.float32) / 255.0
            tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        else:
            # Convert to RGB and normalize
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.input_size, self.input_size))
            image = image.astype(np.float32) / 255.0
            # Normalize with ImageNet stats
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            image = (image - mean) / std
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        
        return tensor.to(self.device)
    
    def predict(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Predict liveness for a single image.
        
        Args:
            image: BGR image (numpy array)
        
        Returns:
            Tuple of (is_live, confidence)
        """
        tensor = self.preprocess(image)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Class 0 = real/live, Class 1 = spoof
            real_prob = probabilities[0][0].item()
            spoof_prob = probabilities[0][1].item()
            
            is_live = real_prob > spoof_prob
            confidence = max(real_prob, spoof_prob)
        
        return is_live, confidence
    
    def get_name(self):
        return f"SimpleCNN({self.input_size}px, {self.device})"


class CNNInferenceSegmenter:
    """
    Adapts the CNN detector to work as a segmentation strategy for comparison.
    
    Since CNN doesn't do grid segmentation, we simulate grid results by
    dividing the image into patches and running CNN on each patch.
    """
    
    def __init__(self, model_path: str = None, patch_size: int = 64, 
                 grayscale: bool = False, stride: int = None):
        self.patch_size = patch_size
        self.stride = stride or patch_size // 2
        self.detector = SimpleCNNDetector(model_path, grayscale)
        self.grayscale = grayscale
    
    def segment(self, image: np.ndarray) -> List:
        """
        Segment image into patches and run CNN on each.
        
        Returns:
            List of simulated GridResult objects (with CNN confidence as variance)
        """
        from liveness_detection.models import GridResult
        
        h, w = image.shape[:2]
        results = []
        
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                
                # Pad if needed
                if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
                    continue
                
                is_live, confidence = self.detector.predict(patch)
                
                # Use confidence as "variance" for compatibility
                # Real patches get higher variance, spoof get lower
                variance = confidence * 1000 if is_live else confidence * 100
                
                results.append(GridResult(
                    variance=variance,
                    position=(x, y),
                    size=(self.patch_size, self.patch_size),
                    is_face_region=True,
                    region=None
                ))
        
        return results
    
    def get_name(self):
        return f"CNNPatch(patch={self.patch_size}, stride={self.stride})"


class CNNFullImageDetector:
    """
    CNN detector that processes the full image at once.
    
    This is the fastest inference method and most realistic for deployment.
    """
    
    def __init__(self, model_path: str = None, grayscale: bool = False,
                 input_size: int = 128, device: str = None):
        self.detector = SimpleCNNDetector(model_path, grayscale, input_size, device)
    
    def detect(self, image: np.ndarray):
        """
        Run CNN on full image for liveness detection.
        
        Returns:
            ProcessingResult: Result compatible with other detectors
        """
        from liveness_detection.models import ProcessingResult
        
        is_live, confidence = self.detector.predict(image)
        
        # Simulate variance metrics for compatibility
        variance = confidence * 1000 if is_live else confidence * 100
        
        return ProcessingResult(
            is_live=is_live,
            confidence=confidence,
            max_variance=variance,
            min_variance=variance,
            avg_variance=variance,
            anomaly_score=1.0 - confidence,
            grid_count=1,
            execution_time=0.0,
            details={"method": "CNN_full_image", "device": self.detector.device}
        )
    
    def get_name(self):
        return f"CNNFullImage({self.detector.input_size}px)"


class LightweightCNNTrainer:
    """
    Training utility for the lightweight CNN.
    
    Handles data loading, training loop, and saving.
    """
    
    def __init__(self, model: LightweightCNN, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    def prepare_dataset(self, dataset_dir: str, batch_size: int = 32,
                       val_split: float = 0.2, grayscale: bool = False,
                       input_size: int = 128):
        """
        Prepare dataset from directories.
        
        Expected structure:
        dataset_dir/
            real/
            spoof/
        """
        import torch.utils.data as data
        from torchvision import transforms
        
        # Data augmentation for training
        if grayscale:
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        # Load dataset
        from torchvision.datasets import ImageFolder
        dataset = ImageFolder(root=dataset_dir, transform=transform)
        
        # Split train/val
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])
        
        self.train_loader = data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)
        self.val_loader = data.DataLoader(val_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=0)
        
        self.class_names = dataset.classes
        print(f"Dataset loaded: {len(dataset)} images")
        print(f"  Train: {train_size}, Val: {val_size}")
        print(f"  Classes: {self.class_names}")
        
        return self.class_names
    
    def train(self, epochs: int = 10, lr: float = 0.001):
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            lr: Learning rate
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        print(f"Training on {self.device} for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                if batch_idx % 10 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(self.train_loader)}, "
                          f"Loss: {loss.item():.4f}")
            
            train_acc = train_correct / train_total
            
            # Validation
            val_loss, val_acc = self._validate()
            
            self.history["train_loss"].append(train_loss / len(self.train_loader))
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss/len(self.train_loader):.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}")
        
        print("Training complete!")
    
    def _validate(self):
        """Run validation."""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        return val_loss / len(self.val_loader), val_correct / val_total
    
    def save(self, path: str, history_path: str = None):
        """Save model and training history."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to: {path}")
        
        if history_path:
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            print(f"History saved to: {history_path}")
    
    def plot_history(self, save_path: str = None):
        """Plot training history."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(self.history["train_loss"], label='Train Loss')
        axes[0].plot(self.history["val_loss"], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].set_title('Loss History')
        
        axes[1].plot(self.history["train_acc"], label='Train Acc')
        axes[1].plot(self.history["val_acc"], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].set_title('Accuracy History')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"History plot saved to: {save_path}")
        
        plt.close()


def get_model_summary(model: nn.Module, input_size: tuple = (1, 3, 128, 128)):
    """
    Print model summary and parameter count.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    return total_params


if __name__ == "__main__":
    # Test model creation
    model = LightweightCNN(num_classes=2, grayscale=False)
    params = get_model_summary(model)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 128, 128)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
