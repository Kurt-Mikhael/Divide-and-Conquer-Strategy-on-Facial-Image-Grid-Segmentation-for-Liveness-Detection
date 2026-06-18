
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_comparison_images(results_dir: str):
    """
    Load all comparison images and organize by label.
    """
    results_path = Path(results_dir)
    real_images = sorted(list(results_path.glob("real_*_comparison.png")))
    spoof_images = sorted(list(results_path.glob("spoof_*_comparison.png")))
    
    return real_images, spoof_images


def create_consolidated_panel(image_paths: list, title: str, save_path: str, max_images: int = 10):
    """
    Create a consolidated panel showing all comparison images.
    
    Args:
        image_paths: List of image paths
        title: Panel title
        save_path: Path to save the figure
        max_images: Maximum number of images to show
    """
    # Limit to max_images
    image_paths = image_paths[:max_images]
    
    n_images = len(image_paths)
    if n_images == 0:
        print(f"No images found for {title}")
        return
    
    # Create figure
    # Each image is 16x6 inches, we'll arrange them in a grid
    # Let's do 2 columns (real/spoof) x 5 rows or similar
    
    n_rows = (n_images + 1) // 2  # 2 per row
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
        
        # Add image number
        ax.set_title(f"Sample #{idx+1}", fontsize=10, pad=5)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Consolidated panel saved: {save_path}")


def create_side_by_side_grid(image_paths: list, label: str, save_path: str):
    """
    Create a 10x2 grid showing original image and method results.
    """
    n_images = len(image_paths)
    if n_images == 0:
        return
    
    fig, axes = plt.subplots(n_images, 3, figsize=(15, n_images * 3))
    fig.suptitle(f"10 {label.upper()} Images - Method Comparison", fontsize=14, fontweight='bold')
    
    if n_images == 1:
        axes = np.array([axes])
    
    for i, img_path in enumerate(image_paths):
        # Load the comparison image
        img = plt.imread(str(img_path))
        
        # The comparison image has 3 columns: Original, D&C, CNN
        # We can display the full image or crop it
        # Since the comparison image is a full figure, let's just display it
        # But that's too tall. Instead, we should re-read the original data
        
        # For simplicity, just show the full comparison image
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Sample #{i+1}", fontsize=10)
        
        # Hide other columns since we're showing the full image in one
        axes[i, 1].axis('off')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_compact_grid(image_paths: list, label: str, save_path: str):
    """
    Create a compact grid showing only the prediction results.
    """
    n_images = len(image_paths)
    if n_images == 0:
        return
    
    # Create 2 rows x 5 columns grid for each method
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f"10 {label.upper()} Images - Divide & Conquer (Top) vs CNN (Bottom)", 
                 fontsize=16, fontweight='bold')
    
    gs = GridSpec(2, 5, figure=fig, hspace=0.2, wspace=0.1)
    
    for i, img_path in enumerate(image_paths):
        if i >= 5:  # Max 5 columns per row
            break
        
        img = plt.imread(str(img_path))
        h, w = img.shape[:2]
        
        # Crop to show only D&C (middle third) and CNN (right third)
        # The comparison image layout: [Original, D&C, CNN]
        # Each is roughly 1/3 of width
        
        # D&C crop (middle section)
        dc_crop = img[:, w//3:2*w//3, :]
        ax1 = fig.add_subplot(gs[0, i])
        ax1.imshow(dc_crop)
        ax1.axis('off')
        ax1.set_title(f"#{i+1} D&C", fontsize=9)
        
        # CNN crop (right section)
        cnn_crop = img[:, 2*w//3:, :]
        ax2 = fig.add_subplot(gs[1, i])
        ax2.imshow(cnn_crop)
        ax2.axis('off')
        ax2.set_title(f"#{i+1} CNN", fontsize=9)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Compact grid saved: {save_path}")


def create_segmentation_grid(results_dir: str, output_dir: str):
    """
    Create comprehensive segmentation visualization grids.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    real_images, spoof_images = load_comparison_images(results_dir)
    
    print(f"Found {len(real_images)} real and {len(spoof_images)} spoof comparison images")
    
    # Create consolidated panels
    create_compact_grid(real_images, "Real", output_dir / "real_10_samples_grid.png")
    create_compact_grid(spoof_images, "Spoof", output_dir / "spoof_10_samples_grid.png")
    
    # Create full panels
    create_consolidated_panel(real_images, "Real Face Images - Complete Comparison", 
                              output_dir / "real_full_panel.png")
    create_consolidated_panel(spoof_images, "Spoof Face Images - Complete Comparison", 
                              output_dir / "spoof_full_panel.png")


def main():
    results_dir = Path("results/comparison")
    output_dir = Path("results/segmentation_grid")
    
    if not results_dir.exists():
        print(f"Comparison results not found at {results_dir}")
        print("Please run comparison_dl.py first")
        return
    
    create_segmentation_grid(str(results_dir), str(output_dir))
    
    print(f"\n{'='*60}")
    print("SEGMENTATION GRID GENERATION COMPLETE")
    print(f"Output: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
