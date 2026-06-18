
import os
import sys
import time
import random
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image

# Import existing modules
from liveness_detection.strategies.variance_calculators import (
    LaplacianVarianceCalculator,
    SobelVarianceCalculator,
    CombinedVarianceCalculator
)
from liveness_detection.segmentation.segmenters import (
    DivideAndConquerSegmenter,
    NaiveFullProcessor,
    SlidingWindowSegmenter
)
from liveness_detection.detection.detectors import (
    ThresholdLivenessDetector,
    AnomalyLivenessDetector,
    EnsembleLivenessDetector,
    VarianceBasedDetector
)
from liveness_detection.processing.pipeline import (
    DatasetProcessor,
    BenchmarkRunner,
    SingleImageAnalyzer
)
from liveness_detection.utils.visualizer import (
    ConsoleLogger,
    Visualizer,
    Timer
)
from liveness_detection.models.dl_cnn import (
    LightweightCNN,
    get_model_summary,
    CNNFullImageDetector
)



class LivenessDataset(data.Dataset):
    
    def __init__(self, dataset_dir: str, transform=None, grayscale: bool = False):
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.grayscale = grayscale
        self.samples = []
        self.class_names = []
        
        class_dirs = sorted([d for d in self.dataset_dir.iterdir() if d.is_dir()])
        for idx, class_dir in enumerate(class_dirs):
            self.class_names.append(class_dir.name)
            image_files = (list(class_dir.glob("*.jpg")) + 
                          list(class_dir.glob("*.png")) + 
                          list(class_dir.glob("*.jpeg")))
            for img_path in image_files:
                self.samples.append((str(img_path), idx))
        
        print(f"Dataset loaded: {len(self.samples)} images")
        print(f"  Classes: {self.class_names}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            image = np.array(Image.open(img_path).convert('RGB'))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if self.transform:
            image = self.transform(image, self.grayscale)
        
        return image, label


def default_transform(image: np.ndarray, grayscale: bool = False, input_size: int = 128):
    if grayscale:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (input_size, input_size))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (input_size, input_size))
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
    
    return torch.from_numpy(image).float()


def augment_transform(image: np.ndarray, grayscale: bool = False, input_size: int = 128):
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)
    
    angle = np.random.uniform(-10, 10)
    if abs(angle) > 0.5:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    return default_transform(image, grayscale, input_size)


def train_cnn_model(dataset_dir: str, output_dir: str, 
                    epochs: int = 5, batch_size: int = 32,
                    lr: float = 0.001, input_size: int = 128,
                    grayscale: bool = False, val_split: float = 0.2):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("STEP 1: LIGHTWEIGHT CNN TRAINING")
    print("="*60)
    
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return None
    
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    print(f"Found classes: {[d.name for d in class_dirs]}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = LightweightCNN(num_classes=2, grayscale=grayscale)
    get_model_summary(model)
    
    print("\nLoading dataset...")
    full_dataset = LivenessDataset(str(dataset_path), transform=None, grayscale=grayscale)
    
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    torch.manual_seed(42)
    train_dataset, val_dataset = data.random_split(full_dataset, [train_size, val_size])
    
    def train_collate(batch):
        images, labels = zip(*batch)
        images = [augment_transform(img.numpy() if isinstance(img, torch.Tensor) else img, 
                                     grayscale, input_size) for img in images]
        images = torch.stack(images)
        labels = torch.tensor(labels)
        return images, labels
    
    def val_collate(batch):
        images, labels = zip(*batch)
        images = [default_transform(img.numpy() if isinstance(img, torch.Tensor) else img, 
                                     grayscale, input_size) for img in images]
        images = torch.stack(images)
        labels = torch.tensor(labels)
        return images, labels
    
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=0, collate_fn=train_collate)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=0, collate_fn=val_collate)
    
    print(f"Dataset: {len(full_dataset)} total")
    print(f"  Train: {train_size}, Val: {val_size}")
    print(f"  Classes: {full_dataset.class_names}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
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
              f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = output_dir / "best_model.pth"
            torch.save(model.state_dict(), best_path)
            print(f"  -> Best model saved! (val_acc={val_acc:.4f})")
    
    total_time = time.time() - total_start
    print(f"\nTraining completed in {total_time:.2f}s")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    final_path = output_dir / "final_model.pth"
    torch.save(model.state_dict(), final_path)
    
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



def run_benchmark(real_dir: str, spoof_dir: str):

    print(f"\n{'='*60}")
    print("STEP 2: BENCHMARK DIVIDE & CONQUER")
    print(f"{'='*60}")
    
    if not Path(real_dir).exists() or not any(Path(real_dir).iterdir()):
        print("Dataset not found. Skipping benchmark.")
        return
    
    methods = [
        (
            "Divide & Conquer (64px)",
            DivideAndConquerSegmenter(min_grid_size=64),
            VarianceBasedDetector(min_variance=100.0, max_variance_ratio=15.0)
        ),
        (
            "Divide & Conquer (32px)",
            DivideAndConquerSegmenter(min_grid_size=32),
            VarianceBasedDetector(min_variance=100.0, max_variance_ratio=15.0)
        ),
        (
            "Naive Full Process",
            NaiveFullProcessor(),
            VarianceBasedDetector(min_variance=100.0, max_variance_ratio=15.0)
        ),
    ]
    
    runner = BenchmarkRunner()
    
    with Timer("Full Benchmark"):
        results = runner.compare_methods(real_dir, spoof_dir, methods)
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    report_path = results_dir / "benchmark_report.txt"
    bench_path = results_dir / "benchmark_comparison.png"
    
    runner.generate_report(results, output_path=str(report_path))
    
    visualizer = Visualizer()
    visualizer.plot_benchmark_comparison(results, save_path=str(bench_path))
    
    print(f"  Saved: {report_path}")
    print(f"  Saved: {bench_path}")
    
    # Complexity analysis
    print(f"\n{'='*60}")
    print("COMPLEXITY ANALYSIS")
    print(f"{'='*60}")
    
    sizes = [64, 128, 256, 512, 1024]
    times = []
    
    segmenter = DivideAndConquerSegmenter(min_grid_size=64)
    
    for size in sizes:
        test_img = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        
        start = time.time()
        segmenter.segment(test_img)
        elapsed = time.time() - start
        
        times.append(elapsed)
        print(f"  Size {size}x{size}: {elapsed:.4f}s")
    
    comp_path = results_dir / "complexity_analysis.png"
    visualizer.plot_complexity_analysis(
        sizes, times, "Divide & Conquer", save_path=str(comp_path)
    )
    print(f"  Saved: {comp_path}")



def select_sample_images(dataset_dir: str, num_samples: int = 10):
    """Select random sample images from dataset."""
    dataset_path = Path(dataset_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    image_files = [
        f for f in dataset_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if len(image_files) < num_samples:
        return [str(f) for f in image_files]
    
    random.seed(42)
    selected = random.sample(image_files, num_samples)
    
    return sorted([str(f) for f in selected])


def run_divide_and_conquer(image_path: str, min_grid_size: int = 64):
    """Run Divide & Conquer algorithm on image."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    variance_calc = LaplacianVarianceCalculator()
    segmenter = DivideAndConquerSegmenter(
        min_grid_size=min_grid_size,
        variance_calculator=variance_calc,
        skip_background=True
    )
    detector = VarianceBasedDetector(min_variance=100.0, max_variance_ratio=15.0)
    analyzer = SingleImageAnalyzer(segmenter, detector)
    
    start = time.time()
    analysis = analyzer.analyze(image_path)
    elapsed = time.time() - start
    
    result = analysis["result"]
    
    return {
        "is_live": result.is_live,
        "confidence": result.confidence,
        "time": elapsed,
        "grid_count": result.grid_count,
        "avg_variance": result.avg_variance,
        "max_variance": result.max_variance,
        "grid_results": analysis["grid_results"],
        "image": img
    }


def run_cnn(image_path: str, model_path: str):
    """Run CNN on image."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    detector = CNNFullImageDetector(model_path=model_path, grayscale=False, input_size=128)
    
    start = time.time()
    result = detector.detect(img)
    elapsed = time.time() - start
    
    return {
        "is_live": result.is_live,
        "confidence": result.confidence,
        "time": elapsed,
        "grid_count": 1,
        "avg_variance": result.avg_variance,
        "max_variance": result.max_variance,
        "image": img
    }


def create_comparison_figure(image_path: str, label: str, 
                             dc_result: dict, cnn_result: dict,
                             save_path: str):
    """Create side-by-side comparison figure for one image."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[3, 1])
    
    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    ax1.set_title(f"Original\nLabel: {label.upper()}", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # D&C Grid Segmentation
    ax2 = fig.add_subplot(gs[0, 1])
    grid_overlay = img.copy()
    
    if dc_result and dc_result.get("grid_results"):
        grid_results = dc_result["grid_results"]
        variances = [g.variance for g in grid_results]
        max_var = max(variances) if variances else 1
        min_var = min(variances) if variances else 0
        var_range = max_var - min_var if max_var != min_var else 1
        
        for grid in grid_results:
            x, y = grid.position
            w, h = grid.size
            normalized = (grid.variance - min_var) / var_range
            color = (int(255 * (1 - normalized)), 0, int(255 * normalized))
            cv2.rectangle(grid_overlay, (x, y), (x + w, y + h), color, 2)
    
    ax2.imshow(cv2.cvtColor(grid_overlay, cv2.COLOR_BGR2RGB))
    
    dc_status = "LIVE" if dc_result["is_live"] else "SPOOF"
    dc_color = "green" if dc_result["is_live"] else "red"
    ax2.set_title(f"Divide & Conquer\nPrediction: {dc_status} (conf={dc_result['confidence']:.3f})\n"
                  f"Time: {dc_result['time']*1000:.1f}ms | Grids: {dc_result['grid_count']}", 
                  fontsize=11, color=dc_color, fontweight='bold')
    ax2.axis('off')
    
    # CNN Result
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img_rgb)
    
    cnn_status = "LIVE" if cnn_result["is_live"] else "SPOOF"
    cnn_color = "green" if cnn_result["is_live"] else "red"
    ax3.set_title(f"Lightweight CNN\nPrediction: {cnn_status} (conf={cnn_result['confidence']:.3f})\n"
                  f"Time: {cnn_result['time']*1000:.1f}ms", 
                  fontsize=11, color=cnn_color, fontweight='bold')
    ax3.axis('off')
    
    # Metrics table
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    
    metrics_data = [
        ["Method", "Prediction", "Confidence", "Time (ms)", "Grid Count", "Avg Variance"],
        ["Divide & Conquer", dc_status, f"{dc_result['confidence']:.4f}", 
         f"{dc_result['time']*1000:.1f}", str(dc_result['grid_count']), 
         f"{dc_result['avg_variance']:.2f}"],
        ["Lightweight CNN", cnn_status, f"{cnn_result['confidence']:.4f}", 
         f"{cnn_result['time']*1000:.1f}", str(cnn_result['grid_count']), 
         f"{cnn_result['avg_variance']:.2f}"]
    ]
    
    table = ax4.table(cellText=metrics_data, cellLoc='center', loc='center',
                      colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    dc_correct = (dc_result["is_live"] and label == "real") or (not dc_result["is_live"] and label == "spoof")
    cnn_correct = (cnn_result["is_live"] and label == "real") or (not cnn_result["is_live"] and label == "spoof")
    
    for i in range(6):
        table[(1, i)].set_facecolor('#C8E6C9' if dc_correct else '#FFCDD2')
        table[(2, i)].set_facecolor('#C8E6C9' if cnn_correct else '#FFCDD2')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_comparison(dataset_dir: str, output_dir: str, model_path: str,num_samples: int = 10, min_grid_size: int = 64):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("STEP 3: DIVIDE & CONQUER vs LIGHTWEIGHT CNN COMPARISON")
    print(f"{'='*80}")
    
    real_dir = Path(dataset_dir) / "real"
    spoof_dir = Path(dataset_dir) / "spoof"
    
    real_samples = select_sample_images(str(real_dir), num_samples)
    spoof_samples = select_sample_images(str(spoof_dir), num_samples)
    
    print(f"\nSelected {len(real_samples)} real and {len(spoof_samples)} spoof images")
    
    results = {
        "real": {"dc": [], "cnn": []},
        "spoof": {"dc": [], "cnn": []}
    }
    
    # Process real images
    print(f"\n{'='*80}")
    print("PROCESSING REAL IMAGES")
    print(f"{'='*80}")
    
    for i, img_path in enumerate(real_samples, 1):
        print(f"\n[{i}/{len(real_samples)}] {Path(img_path).name}")
        
        dc_result = run_divide_and_conquer(img_path, min_grid_size)
        if dc_result:
            results["real"]["dc"].append(dc_result)
            print(f"  D&C: {dc_result['time']*1000:.1f}ms, grids={dc_result['grid_count']}, live={dc_result['is_live']}")
        
        cnn_result = run_cnn(img_path, model_path)
        if cnn_result:
            results["real"]["cnn"].append(cnn_result)
            print(f"  CNN: {cnn_result['time']*1000:.1f}ms, live={cnn_result['is_live']}")
        
        if dc_result and cnn_result:
            save_path = output_dir / f"real_{i:02d}_comparison.png"
            create_comparison_figure(img_path, "real", dc_result, cnn_result, str(save_path))
    
    # Process spoof images
    print(f"\n{'='*80}")
    print("PROCESSING SPOOF IMAGES")
    print(f"{'='*80}")
    
    for i, img_path in enumerate(spoof_samples, 1):
        print(f"\n[{i}/{len(spoof_samples)}] {Path(img_path).name}")
        
        dc_result = run_divide_and_conquer(img_path, min_grid_size)
        if dc_result:
            results["spoof"]["dc"].append(dc_result)
            print(f"  D&C: {dc_result['time']*1000:.1f}ms, grids={dc_result['grid_count']}, live={dc_result['is_live']}")
        
        cnn_result = run_cnn(img_path, model_path)
        if cnn_result:
            results["spoof"]["cnn"].append(cnn_result)
            print(f"  CNN: {cnn_result['time']*1000:.1f}ms, live={cnn_result['is_live']}")
        
        if dc_result and cnn_result:
            save_path = output_dir / f"spoof_{i:02d}_comparison.png"
            create_comparison_figure(img_path, "spoof", dc_result, cnn_result, str(save_path))
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Generate summary report
    generate_summary_report(results, metrics, str(output_dir))
    
    return results, metrics


def calculate_metrics(results: dict):
    metrics = {
        "dc": {"correct": 0, "total": 0, "time": [], "real_correct": 0, "spoof_correct": 0},
        "cnn": {"correct": 0, "total": 0, "time": [], "real_correct": 0, "spoof_correct": 0}
    }
    
    for dc_res in results["real"]["dc"]:
        metrics["dc"]["total"] += 1
        metrics["dc"]["time"].append(dc_res["time"])
        if dc_res["is_live"]:
            metrics["dc"]["correct"] += 1
            metrics["dc"]["real_correct"] += 1
    
    for cnn_res in results["real"]["cnn"]:
        metrics["cnn"]["total"] += 1
        metrics["cnn"]["time"].append(cnn_res["time"])
        if cnn_res["is_live"]:
            metrics["cnn"]["correct"] += 1
            metrics["cnn"]["real_correct"] += 1
    
    for dc_res in results["spoof"]["dc"]:
        metrics["dc"]["total"] += 1
        metrics["dc"]["time"].append(dc_res["time"])
        if not dc_res["is_live"]:
            metrics["dc"]["correct"] += 1
            metrics["dc"]["spoof_correct"] += 1
    
    for cnn_res in results["spoof"]["cnn"]:
        metrics["cnn"]["total"] += 1
        metrics["cnn"]["time"].append(cnn_res["time"])
        if not cnn_res["is_live"]:
            metrics["cnn"]["correct"] += 1
            metrics["cnn"]["spoof_correct"] += 1
    
    for method in ["dc", "cnn"]:
        total = metrics[method]["total"]
        if total > 0:
            metrics[method]["accuracy"] = metrics[method]["correct"] / total
            metrics[method]["avg_time"] = np.mean(metrics[method]["time"])
            metrics[method]["std_time"] = np.std(metrics[method]["time"])
        else:
            metrics[method]["accuracy"] = 0
            metrics[method]["avg_time"] = 0
            metrics[method]["std_time"] = 0
    
    return metrics


def generate_summary_report(results: dict, metrics: dict, output_dir: str):
    output_dir = Path(output_dir)
    
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Method':<20} {'Accuracy':<12} {'Avg Time (ms)':<15} {'Std Time (ms)':<15}")
    print("-" * 65)
    
    for method_name, method_key in [("Divide & Conquer", "dc"), ("Lightweight CNN", "cnn")]:
        m = metrics[method_key]
        print(f"{method_name:<20} {m['accuracy']*100:.1f}%{'':<6} "
              f"{m['avg_time']*1000:.2f}{'':<9} {m['std_time']*1000:.2f}")
    
    print(f"\n{'='*80}")
    print("ACCURACY BREAKDOWN")
    print(f"{'='*80}")
    
    for method_name, method_key in [("Divide & Conquer", "dc"), ("Lightweight CNN", "cnn")]:
        m = metrics[method_key]
        real_total = len(results["real"][method_key])
        spoof_total = len(results["spoof"][method_key])
        
        real_acc = m["real_correct"] / real_total * 100 if real_total > 0 else 0
        spoof_acc = m["spoof_correct"] / spoof_total * 100 if spoof_total > 0 else 0
        
        print(f"\n{method_name}:")
        print(f"  Real Accuracy:  {m['real_correct']}/{real_total} ({real_acc:.1f}%)")
        print(f"  Spoof Accuracy: {m['spoof_correct']}/{spoof_total} ({spoof_acc:.1f}%)")
    
    # Save to file
    report_path = output_dir / "comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DIVIDE & CONQUER vs LIGHTWEIGHT CNN COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        f.write("METRICS SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Method':<20} {'Accuracy':<12} {'Avg Time (ms)':<15} {'Std Time (ms)':<15}\n")
        f.write("-"*80 + "\n")
        
        for method_name, method_key in [("Divide & Conquer", "dc"), ("Lightweight CNN", "cnn")]:
            m = metrics[method_key]
            f.write(f"{method_name:<20} {m['accuracy']*100:.1f}%{'':<6} "
                   f"{m['avg_time']*1000:.2f}{'':<9} {m['std_time']*1000:.2f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("ACCURACY BREAKDOWN\n")
        f.write("="*80 + "\n")
        
        for method_name, method_key in [("Divide & Conquer", "dc"), ("Lightweight CNN", "cnn")]:
            m = metrics[method_key]
            real_total = len(results["real"][method_key])
            spoof_total = len(results["spoof"][method_key])
            
            real_acc = m["real_correct"] / real_total * 100 if real_total > 0 else 0
            spoof_acc = m["spoof_correct"] / spoof_total * 100 if spoof_total > 0 else 0
            
            f.write(f"\n{method_name}:\n")
            f.write(f"  Real Accuracy:  {m['real_correct']}/{real_total} ({real_acc:.1f}%)\n")
            f.write(f"  Spoof Accuracy: {m['spoof_correct']}/{spoof_total} ({spoof_acc:.1f}%)\n")
    
    print(f"\nReport saved to: {report_path}")
    
    # Generate comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = ["Divide & Conquer", "Lightweight CNN"]
    method_keys = ["dc", "cnn"]
    
    accuracies = [metrics[k]["accuracy"] * 100 for k in method_keys]
    colors = ['steelblue', 'coral']
    axes[0, 0].bar(methods, accuracies, color=colors)
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Overall Accuracy Comparison')
    axes[0, 0].set_ylim(0, 100)
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    avg_times = [metrics[k]["avg_time"] * 1000 for k in method_keys]
    std_times = [metrics[k]["std_time"] * 1000 for k in method_keys]
    axes[0, 1].bar(methods, avg_times, yerr=std_times, color=colors, capsize=5)
    axes[0, 1].set_ylabel('Avg Time (ms)')
    axes[0, 1].set_title('Execution Time Comparison')
    for i, v in enumerate(avg_times):
        axes[0, 1].text(i, v + std_times[i] + 2, f'{v:.1f}ms', ha='center', fontweight='bold')
    
    real_accs = []
    spoof_accs = []
    for k in method_keys:
        real_total = len(results["real"][k])
        spoof_total = len(results["spoof"][k])
        real_accs.append(metrics[k]["real_correct"] / real_total * 100 if real_total > 0 else 0)
        spoof_accs.append(metrics[k]["spoof_correct"] / spoof_total * 100 if spoof_total > 0 else 0)
    
    x = np.arange(len(methods))
    width = 0.35
    axes[1, 0].bar(x - width/2, real_accs, width, label='Real', color='lightgreen')
    axes[1, 0].bar(x + width/2, spoof_accs, width, label='Spoof', color='lightcoral')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Per-Class Accuracy')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(methods)
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 100)
    
    for i, (method, key) in enumerate(zip(methods, method_keys)):
        times = [t * 1000 for t in metrics[key]["time"]]
        axes[1, 1].hist(times, bins=10, alpha=0.6, label=method, color=colors[i])
    
    axes[1, 1].set_xlabel('Time (ms)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Time Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    chart_path = output_dir / "comparison_charts.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison charts saved to: {chart_path}")
    
    json_path = output_dir / "metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics JSON saved to: {json_path}")




def generate_segmentation_grid(results_dir: str, output_dir: str):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = Path(results_dir)
    real_images = sorted(list(results_path.glob("real_*_comparison.png")))
    spoof_images = sorted(list(results_path.glob("spoof_*_comparison.png")))
    
    print(f"\n{'='*60}")
    print("STEP 4: GENERATE SEGMENTATION GRID")
    print(f"{'='*60}")
    print(f"Found {len(real_images)} real and {len(spoof_images)} spoof comparison images")
    
    # Compact grid - 5 columns per row
    def create_compact_grid(image_paths: list, label: str, save_path: str):
        n_images = len(image_paths)
        if n_images == 0:
            return
        
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f"10 {label.upper()} Images - Divide & Conquer (Top) vs CNN (Bottom)", 
                     fontsize=16, fontweight='bold')
        
        gs = GridSpec(2, 5, figure=fig, hspace=0.2, wspace=0.1)
        
        for i, img_path in enumerate(image_paths):
            if i >= 5:
                break
            
            img = plt.imread(str(img_path))
            h, w = img.shape[:2]
            
            dc_crop = img[:, w//3:2*w//3, :]
            ax1 = fig.add_subplot(gs[0, i])
            ax1.imshow(dc_crop)
            ax1.axis('off')
            ax1.set_title(f"#{i+1} D&C", fontsize=9)
            
            cnn_crop = img[:, 2*w//3:, :]
            ax2 = fig.add_subplot(gs[1, i])
            ax2.imshow(cnn_crop)
            ax2.axis('off')
            ax2.set_title(f"#{i+1} CNN", fontsize=9)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Compact grid saved: {save_path}")
    
    # Full panel - 2 columns
    def create_full_panel(image_paths: list, title: str, save_path: str):
        image_paths = image_paths[:10]
        n_images = len(image_paths)
        if n_images == 0:
            print(f"No images found for {title}")
            return
        
        n_rows = (n_images + 1) // 2
        fig_height = n_rows * 5
        fig_width = 16
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        gs = GridSpec(n_rows, 2, figure=fig, hspace=0.3, wspace=0.1)
        
        for idx, img_path in enumerate(image_paths):
            row = idx // 2
            col = idx % 2
            
            ax = fig.add_subplot(gs[row, col])
            img = plt.imread(str(img_path))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Sample #{idx+1}", fontsize=10, pad=5)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Full panel saved: {save_path}")
    
    create_compact_grid(real_images, "Real", output_dir / "real_10_samples_grid.png")
    create_compact_grid(spoof_images, "Spoof", output_dir / "spoof_10_samples_grid.png")
    create_full_panel(real_images, "Real Face Images - Complete Comparison", output_dir / "real_full_panel.png")
    create_full_panel(spoof_images, "Spoof Face Images - Complete Comparison", output_dir / "spoof_full_panel.png")



def main():
    print("="*80)
    print("LIVENESS DETECTION SYSTEM")
    print("Divide & Conquer vs Lightweight CNN")
    print("="*80)
    
    # Detect paths
    script_dir = Path(__file__).parent
    dataset_dir = script_dir / "datasets"
    model_dir = script_dir / "models" / "saved"
    
    # Fallback
    if not dataset_dir.exists():
        dataset_dir = Path("liveness_detection/datasets")
        model_dir = Path("liveness_detection/models/saved")
    
    model_path = model_dir / "best_model.pth"
    
    print(f"\nDataset: {dataset_dir}")
    print(f"Model: {model_path}")
    
    # Check if dataset exists
    if not dataset_dir.exists():
        print("\nERROR: Dataset not found!")
        print(f"Expected at: {dataset_dir}")
        return
    
    try:

        if not model_path.exists():
            print("\nModel not found. Starting training...")
            trained_model = train_cnn_model(
                dataset_dir=str(dataset_dir),
                output_dir=str(model_dir),
                epochs=5,
                batch_size=32,
                lr=0.001,
                input_size=128,
                grayscale=False,
                val_split=0.2
            )
            if not trained_model:
                print("Training failed! Exiting.")
                return
            model_path = Path(trained_model)
        else:
            print(f"\nModel found: {model_path}")
            print("Skipping training. To retrain, delete the model file.")
        

        real_dir = dataset_dir / "real"
        spoof_dir = dataset_dir / "spoof"
        
        if real_dir.exists() and spoof_dir.exists():
            run_benchmark(str(real_dir), str(spoof_dir))
        else:
            print("\nDataset directories not found. Skipping benchmark.")
        

        results, metrics = run_comparison(
            dataset_dir=str(dataset_dir),
            output_dir="results/comparison",
            model_path=str(model_path),
            num_samples=10,
            min_grid_size=64
        )
        

        generate_segmentation_grid(
            results_dir="results/comparison",
            output_dir="results/segmentation_grid"
        )
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("ALL PROCESSES COMPLETE")
    print("="*80)
    print("\nResults saved to:")
    print("  results/benchmark_report.txt")
    print("  results/benchmark_comparison.png")
    print("  results/complexity_analysis.png")
    print("  results/comparison/ (20 comparison images + charts)")
    print("  results/segmentation_grid/ (4 grid visualizations)")
    print("="*80)


if __name__ == "__main__":
    main()
