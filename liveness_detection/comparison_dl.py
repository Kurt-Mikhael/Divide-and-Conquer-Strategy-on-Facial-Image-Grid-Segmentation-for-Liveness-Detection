
import os
import sys
import time
import json
import random
from pathlib import Path
from collections import defaultdict

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch

# Import existing modules
from liveness_detection.segmentation.segmenters import DivideAndConquerSegmenter
from liveness_detection.strategies.variance_calculators import LaplacianVarianceCalculator
from liveness_detection.detection.detectors import VarianceBasedDetector
from liveness_detection.processing.pipeline import SingleImageAnalyzer
from liveness_detection.models.dl_cnn import SimpleCNNDetector, CNNFullImageDetector
from liveness_detection.utils.visualizer import Visualizer


def select_sample_images(dataset_dir: str, num_samples: int = 10):

    dataset_path = Path(dataset_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    image_files = [
        f for f in dataset_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if len(image_files) < num_samples:
        return [str(f) for f in image_files]
    
    # Set seed for reproducibility
    random.seed(42)
    selected = random.sample(image_files, num_samples)
    
    return sorted([str(f) for f in selected])


def run_divide_and_conquer(image_path: str, min_grid_size: int = 64):

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


def create_comparison_figure(image_path: str, label: str, dc_result: dict, cnn_result: dict,save_path: str):

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
            color = (
                int(255 * (1 - normalized)),
                0,
                int(255 * normalized)
            )
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
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style correct predictions
    dc_correct = (dc_result["is_live"] and label == "real") or (not dc_result["is_live"] and label == "spoof")
    cnn_correct = (cnn_result["is_live"] and label == "real") or (not cnn_result["is_live"] and label == "spoof")
    
    for i in range(6):
        table[(1, i)].set_facecolor('#C8E6C9' if dc_correct else '#FFCDD2')
        table[(2, i)].set_facecolor('#C8E6C9' if cnn_correct else '#FFCDD2')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_comparison(dataset_dir: str, output_dir: str, model_path: str,
                   num_samples: int = 10, min_grid_size: int = 64):
    """
    Run full comparison between D&C and CNN.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("DIVIDE & CONQUER vs LIGHTWEIGHT CNN COMPARISON")
    print("="*80)
    
    # Select samples
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
        
        # D&C
        dc_result = run_divide_and_conquer(img_path, min_grid_size)
        if dc_result:
            results["real"]["dc"].append(dc_result)
            print(f"  D&C: {dc_result['time']*1000:.1f}ms, grids={dc_result['grid_count']}, "
                  f"live={dc_result['is_live']}")
        
        # CNN
        cnn_result = run_cnn(img_path, model_path)
        if cnn_result:
            results["real"]["cnn"].append(cnn_result)
            print(f"  CNN: {cnn_result['time']*1000:.1f}ms, "
                  f"live={cnn_result['is_live']}")
        
        # Save comparison figure
        if dc_result and cnn_result:
            save_path = output_dir / f"real_{i:02d}_comparison.png"
            create_comparison_figure(img_path, "real", dc_result, cnn_result, str(save_path))
    
    # Process spoof images
    print(f"\n{'='*80}")
    print("PROCESSING SPOOF IMAGES")
    print(f"{'='*80}")
    
    for i, img_path in enumerate(spoof_samples, 1):
        print(f"\n[{i}/{len(spoof_samples)}] {Path(img_path).name}")
        
        # D&C
        dc_result = run_divide_and_conquer(img_path, min_grid_size)
        if dc_result:
            results["spoof"]["dc"].append(dc_result)
            print(f"  D&C: {dc_result['time']*1000:.1f}ms, grids={dc_result['grid_count']}, "
                  f"live={dc_result['is_live']}")
        
        # CNN
        cnn_result = run_cnn(img_path, model_path)
        if cnn_result:
            results["spoof"]["cnn"].append(cnn_result)
            print(f"  CNN: {cnn_result['time']*1000:.1f}ms, "
                  f"live={cnn_result['is_live']}")
        
        # Save comparison figure
        if dc_result and cnn_result:
            save_path = output_dir / f"spoof_{i:02d}_comparison.png"
            create_comparison_figure(img_path, "spoof", dc_result, cnn_result, str(save_path))
    
    return results


def calculate_metrics(results: dict):

    metrics = {
        "dc": {"correct": 0, "total": 0, "time": [], "real_correct": 0, "spoof_correct": 0},
        "cnn": {"correct": 0, "total": 0, "time": [], "real_correct": 0, "spoof_correct": 0}
    }
    
    # Real images: is_live should be True
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
    
    # Spoof images: is_live should be False
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
    
    # Calculate percentages
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
    
    # Print report
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
    
    # Accuracy comparison
    accuracies = [metrics[k]["accuracy"] * 100 for k in method_keys]
    colors = ['steelblue', 'coral']
    axes[0, 0].bar(methods, accuracies, color=colors)
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Overall Accuracy Comparison')
    axes[0, 0].set_ylim(0, 100)
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Time comparison
    avg_times = [metrics[k]["avg_time"] * 1000 for k in method_keys]
    std_times = [metrics[k]["std_time"] * 1000 for k in method_keys]
    axes[0, 1].bar(methods, avg_times, yerr=std_times, color=colors, capsize=5)
    axes[0, 1].set_ylabel('Avg Time (ms)')
    axes[0, 1].set_title('Execution Time Comparison')
    for i, v in enumerate(avg_times):
        axes[0, 1].text(i, v + std_times[i] + 2, f'{v:.1f}ms', ha='center', fontweight='bold')
    
    # Real vs Spoof accuracy
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
    
    # Time distribution
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
    
    # Save metrics JSON
    json_path = output_dir / "metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics JSON saved to: {json_path}")


def main():
    # Paths
    script_dir = Path(__file__).parent
    dataset_dir = script_dir.parent / "datasets"
    model_path = script_dir.parent / "models" / "saved" / "best_model.pth"
    output_dir = Path("results") / "comparison"
    
    # Fallback
    if not dataset_dir.exists():
        dataset_dir = Path("liveness_detection/datasets")
        model_path = Path("liveness_detection/models/saved/best_model.pth")
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first using train_dl.py")
        return
    
    print(f"Dataset: {dataset_dir}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    
    # Run comparison
    results = run_comparison(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        model_path=str(model_path),
        num_samples=10,
        min_grid_size=64
    )
    
    metrics = calculate_metrics(results)
    
    # Generate summary
    generate_summary_report(results, metrics, str(output_dir))
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
