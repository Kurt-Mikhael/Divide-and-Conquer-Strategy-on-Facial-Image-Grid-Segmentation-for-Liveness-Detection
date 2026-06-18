"""
Utility modules for logging, timing, and visualization.
"""
import csv
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - always save to file
import matplotlib.pyplot as plt
from liveness_detection.interfaces import IBenchmarkLogger
from liveness_detection.models import GridResult, ProcessingResult


class ConsoleLogger(IBenchmarkLogger):
    """Simple console-based logger.
    
    Single Responsibility: Logs execution metrics to console.
    """
    
    def __init__(self):
        self.logs = []
    
    def log_execution(self, image_path: str, execution_time: float, 
                     grid_count: int, result: str) -> None:
        """Log execution to console."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "image": image_path,
            "time": execution_time,
            "grids": grid_count,
            "result": result
        }
        self.logs.append(log_entry)
        print(f"[{log_entry['timestamp']}] {Path(image_path).name}: "
              f"{execution_time:.4f}s | {grid_count} grids | {result}")
    
    def log_comparison(self, method_name: str, total_time: float, 
                      avg_time: float, accuracy: float) -> None:
        """Log comparison results."""
        print(f"\nComparison: {method_name}")
        print(f"  Total Time: {total_time:.4f}s")
        print(f"  Avg Time:   {avg_time:.4f}s")
        print(f"  Accuracy:   {accuracy:.4f}")
    
    def save_results(self, output_path: str) -> None:
        """Save logs to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.logs, f, indent=2)
        print(f"Logs saved to: {output_path}")


class CSVLogger(IBenchmarkLogger):
    """CSV-based logger for structured data export.
    
    Useful for importing results into analysis tools.
    """
    
    def __init__(self):
        self.execution_logs = []
        self.comparison_logs = []
    
    def log_execution(self, image_path: str, execution_time: float, 
                     grid_count: int, result: str) -> None:
        """Log execution to CSV buffer."""
        self.execution_logs.append({
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "execution_time": execution_time,
            "grid_count": grid_count,
            "result": result
        })
    
    def log_comparison(self, method_name: str, total_time: float, 
                      avg_time: float, accuracy: float) -> None:
        """Log comparison to CSV buffer."""
        self.comparison_logs.append({
            "method_name": method_name,
            "total_time": total_time,
            "avg_time": avg_time,
            "accuracy": accuracy
        })
    
    def save_results(self, output_path: str) -> None:
        """Save results to CSV files."""
        base_path = Path(output_path)
        
        # Save execution logs
        exec_path = base_path.with_suffix('.execution.csv')
        with open(exec_path, 'w', newline='') as f:
            if self.execution_logs:
                writer = csv.DictWriter(f, fieldnames=self.execution_logs[0].keys())
                writer.writeheader()
                writer.writerows(self.execution_logs)
        
        # Save comparison logs
        comp_path = base_path.with_suffix('.comparison.csv')
        with open(comp_path, 'w', newline='') as f:
            if self.comparison_logs:
                writer = csv.DictWriter(f, fieldnames=self.comparison_logs[0].keys())
                writer.writeheader()
                writer.writerows(self.comparison_logs)
        
        print(f"CSV logs saved to: {exec_path} and {comp_path}")


class Timer:
    """Context manager for timing code blocks.
    
    Usage:
        with Timer("operation") as t:
            # do something
        print(f"Took {t.elapsed:.4f}s")
    """
    
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.elapsed = 0.0
        self._start = None
    
    def __enter__(self):
        self._start = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self._start
        print(f"[{self.name}] Elapsed: {self.elapsed:.4f}s")


class Visualizer:
    """Visualization utilities for liveness detection results.
    
    Single Responsibility: Handles all visualization and plotting.
    """
    
    def __init__(self, figsize: tuple = (15, 10)):
        self.figsize = figsize
    
    def visualize_grid_segmentation(self, 
                                   image: np.ndarray, 
                                   grid_results: List[GridResult],
                                   save_path: str = None):
        """Visualize grid segmentation on image.
        
        Args:
            image: Original image
            grid_results: Grid analysis results
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Original image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Grid visualization
        grid_overlay = image.copy()
        
        # Normalize variances for coloring
        variances = [g.variance for g in grid_results]
        max_var = max(variances) if variances else 1
        min_var = min(variances) if variances else 0
        var_range = max_var - min_var if max_var != min_var else 1
        
        for grid in grid_results:
            x, y = grid.position
            w, h = grid.size
            
            # Color based on variance (red = high, blue = low)
            # OpenCV uses BGR format: (Blue, Green, Red)
            normalized = (grid.variance - min_var) / var_range
            color = (
                int(255 * (1 - normalized)),  # B = Blue (low variance)
                0,                            # G = Green
                int(255 * normalized)         # R = Red (high variance)
            )
            
            cv2.rectangle(grid_overlay, (x, y), (x + w, y + h), color, 2)
            
            # Add variance text for larger grids
            if w > 40 and h > 40:
                text = f"{grid.variance:.1f}"
                cv2.putText(grid_overlay, text, (x + 2, y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Add legend text
        legend_text = "RED=High Var (Real) | BLUE=Low Var (Spoof/Flat)"
        cv2.putText(grid_overlay, legend_text, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(grid_overlay, legend_text, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        axes[1].imshow(cv2.cvtColor(grid_overlay, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Grid Segmentation ({len(grid_results)} grids)')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.close()
    
    def plot_variance_distribution(self,
                                  grid_results: List[GridResult],
                                  title: str = "Variance Distribution",
                                  save_path: str = None):
        """Plot variance distribution histogram.
        
        Args:
            grid_results: Grid results
            title: Plot title
            save_path: Optional save path
        """
        variances = [g.variance for g in grid_results]
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Histogram
        axes[0].hist(variances, bins=30, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Variance')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(title)
        axes[0].axvline(np.mean(variances), color='red', linestyle='--', label=f'Mean: {np.mean(variances):.1f}')
        axes[0].legend()
        
        # Box plot
        axes[1].boxplot(variances, vert=True)
        axes[1].set_ylabel('Variance')
        axes[1].set_title('Variance Box Plot')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def plot_benchmark_comparison(self,
                                 results: dict,
                                 save_path: str = None):
        """Plot benchmark comparison between methods.
        
        Args:
            results: Dict of method name to BenchmarkResult
            save_path: Optional save path
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        methods = list(results.keys())
        
        # Execution time comparison
        avg_times = [results[m].avg_time for m in methods]
        axes[0, 0].bar(methods, avg_times, color=['steelblue', 'coral', 'lightgreen'])
        axes[0, 0].set_ylabel('Avg Time (s)')
        axes[0, 0].set_title('Execution Time Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Accuracy comparison
        accuracies = [results[m].accuracy * 100 for m in methods]
        axes[0, 1].bar(methods, accuracies, color=['steelblue', 'coral', 'lightgreen'])
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylim(0, 100)
        
        # Grids processed
        total_grids = [results[m].total_grids for m in methods]
        axes[1, 0].bar(methods, total_grids, color=['steelblue', 'coral', 'lightgreen'])
        axes[1, 0].set_ylabel('Total Grids')
        axes[1, 0].set_title('Grids Processed')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # F1 Score
        f1_scores = [results[m].f1_score for m in methods]
        axes[1, 1].bar(methods, f1_scores, color=['steelblue', 'coral', 'lightgreen'])
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('F1 Score Comparison')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Benchmark visualization saved to: {save_path}")
        
        plt.close()
    
    def plot_complexity_analysis(self,
                               image_sizes: List[int],
                               execution_times: List[float],
                               method_name: str,
                               save_path: str = None):
        """Plot time complexity analysis.
        
        Args:
            image_sizes: List of image sizes (N x N)
            execution_times: Corresponding execution times
            method_name: Name of the method
            save_path: Optional save path
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Raw plot
        axes[0].plot(image_sizes, execution_times, 'o-', color='steelblue', linewidth=2)
        axes[0].set_xlabel('Image Size (N x N pixels)')
        axes[0].set_ylabel('Execution Time (s)')
        axes[0].set_title(f'{method_name}: Execution Time vs Image Size')
        axes[0].grid(True, alpha=0.3)
        
        # Log-log plot for complexity analysis
        log_sizes = np.log2(image_sizes)
        log_times = np.log2(execution_times)
        
        axes[1].plot(log_sizes, log_times, 'o-', color='coral', linewidth=2)
        
        # Fit line to determine slope
        if len(log_sizes) > 1:
            coeffs = np.polyfit(log_sizes, log_times, 1)
            slope = coeffs[0]
            fit_line = np.poly1d(coeffs)
            axes[1].plot(log_sizes, fit_line(log_sizes), '--', 
                        color='green', label=f'Slope: {slope:.2f}')
        
        axes[1].set_xlabel('log2(Image Size)')
        axes[1].set_ylabel('log2(Execution Time)')
        axes[1].set_title('Complexity Analysis (log-log)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
        
        if len(log_sizes) > 1:
            print(f"\nComplexity Analysis:")
            print(f"  Estimated slope: {slope:.2f}")
            print(f"  Expected O(N log N) slope: ~1.0-2.0")
            print(f"  Expected O(N^2) slope: ~2.0")
            print(f"  The measured slope indicates the actual complexity.")
