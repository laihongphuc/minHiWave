"""
HiWave Image Comparison Tool

This script provides an interactive tool to compare low-resolution and super-resolution images,
allowing users to select regions and view detailed comparisons.
"""

from PIL import Image
import torch 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import os

def load_images(lr_path="content/initial_generated_image_1.png", 
                sr_path="content/high_resolution_landscape_1.png"):
    """
    Load low-resolution and super-resolution images for comparison.
    
    Args:
        lr_path: Path to low-resolution image
        sr_path: Path to super-resolution image
    
    Returns:
        tuple: (lr_data, sr_data, scaling_factor)
    """
    # Check if files exist
    if not os.path.exists(lr_path):
        print(f"Warning: Low-resolution image not found at {lr_path}")
        print("Please ensure you have generated images in the content/ folder")
        return None, None, None
    
    if not os.path.exists(sr_path):
        print(f"Warning: Super-resolution image not found at {sr_path}")
        print("Please ensure you have generated images in the content/ folder")
        return None, None, None
    
    # Load images
    lr_img = Image.open(lr_path)
    sr_img = Image.open(sr_path)
    
    lr_data = np.array(lr_img)
    sr_data = np.array(sr_img)
    
    # Calculate upscaling factor
    height_lr, width_lr = lr_data.shape[:2]
    height_sr, width_sr = sr_data.shape[:2]
    k = width_sr / width_lr  # Assumes uniform scaling
    
    print(f"Loaded images:")
    print(f"  Low-resolution: {width_lr}x{height_lr}")
    print(f"  Super-resolution: {width_sr}x{height_sr}")
    print(f"  Scaling factor: {k:.2f}x")
    
    return lr_data, sr_data, k

def create_comparison_view(lr_data, sr_data, k):
    """
    Create an interactive comparison view of the images.
    
    Args:
        lr_data: Low-resolution image data
        sr_data: Super-resolution image data
        k: Scaling factor between images
    """
    # Create figure with two subplots
    fig, (ax_lr, ax_sr) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Display images
    ax_lr.imshow(lr_data)
    ax_sr.imshow(sr_data)
    
    # Set titles and remove axes
    ax_lr.set_title('Low Resolution (Original)', fontsize=14, fontweight='bold')
    ax_sr.set_title('Super Resolution (HiWave Enhanced)', fontsize=14, fontweight='bold')
    ax_lr.axis('off')
    ax_sr.axis('off')
    
    # Add inset axes for zoomed-in views
    ax_lr_inset = ax_lr.inset_axes([0.6, 0.6, 0.35, 0.35])
    ax_sr_inset = ax_sr.inset_axes([0.6, 0.6, 0.35, 0.35])
    
    ax_lr_inset.imshow(lr_data)
    ax_sr_inset.imshow(sr_data)
    ax_lr_inset.axis('off')
    ax_sr_inset.axis('off')
    
    # Store rectangles for selection indication
    rectangles = {'lr': None, 'sr': None}
    
    def on_select(eclick, erelease):
        """Handle region selection and update zoomed views."""
        # Remove previous rectangles if they exist
        for key in rectangles:
            if rectangles[key] is not None:
                rectangles[key].remove()
                rectangles[key] = None
        
        # Get selection coordinates
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        
        # Update inset views
        ax_lr_inset.set_xlim(x1, x2)
        ax_lr_inset.set_ylim(y2, y1)  # Flip y-axis for correct orientation
        ax_sr_inset.set_xlim(k * x1, k * x2)
        ax_sr_inset.set_ylim(k * y2, k * y1)  # Scale and flip for SR
        
        # Draw rectangles to indicate selected regions
        rectangles['lr'] = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                       edgecolor='red', facecolor='none', linewidth=2)
        rectangles['sr'] = plt.Rectangle((k * x1, k * y1), k * (x2 - x1), k * (y2 - y1), 
                                       edgecolor='red', facecolor='none', linewidth=2)
        ax_lr.add_patch(rectangles['lr'])
        ax_sr.add_patch(rectangles['sr'])
        
        fig.canvas.draw()
        
        # Crop the selected regions
        row_start = max(0, int(np.floor(y1)))
        row_end = min(lr_data.shape[0], int(np.ceil(y2)))
        col_start = max(0, int(np.floor(x1)))
        col_end = min(lr_data.shape[1], int(np.ceil(x2)))
        
        row_start_sr = max(0, int(np.floor(k * y1)))
        row_end_sr = min(sr_data.shape[0], int(np.ceil(k * y2)))
        col_start_sr = max(0, int(np.floor(k * x1)))
        col_end_sr = min(sr_data.shape[1], int(np.ceil(k * x2)))
        
        if lr_data.ndim == 3:
            lr_cropped = lr_data[row_start:row_end, col_start:col_end, :]
            sr_cropped = sr_data[row_start_sr:row_end_sr, col_start_sr:col_end_sr, :]
        else:
            lr_cropped = lr_data[row_start:row_end, col_start:col_end]
            sr_cropped = sr_data[row_start_sr:row_end_sr, col_start_sr:col_end_sr]
        
        # Get target size from sr_cropped
        target_height, target_width = sr_cropped.shape[0], sr_cropped.shape[1]
        
        # Resize lr_cropped to match sr_cropped size
        if lr_cropped.ndim == 2:
            lr_pil = Image.fromarray(lr_cropped, mode='L')
        else:
            lr_pil = Image.fromarray(lr_cropped, mode='RGB')
        lr_resized = lr_pil.resize((target_width, target_height), Image.BICUBIC)
        
        # Save the resized LR and SR cropped images
        lr_resized.save('content/lr_zoomed_1.png')
        if sr_cropped.ndim == 2:
            sr_pil = Image.fromarray(sr_cropped, mode='L')
        else:
            sr_pil = Image.fromarray(sr_cropped, mode='RGB')
        sr_pil.save('content/sr_zoomed_1.png')
        
        print(f"Selected region saved:")
        print(f"  Low-resolution zoom: content/lr_zoomed_1.png")
        print(f"  Super-resolution zoom: content/sr_zoomed_1.png")
    
    # Add RectangleSelector to LR image for interactive selection
    selector = RectangleSelector(ax_lr, on_select, useblit=True, button=[1], 
                               minspanx=5, minspany=5, spancoords='pixels', 
                               interactive=True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the comparison tool."""
    print("HiWave Image Comparison Tool")
    print("=" * 40)
    print("Instructions:")
    print("1. Click and drag on the left image to select a region")
    print("2. The right image will show the corresponding scaled region")
    print("3. Zoomed views will appear in the insets")
    print("4. Selected regions will be saved to content/ folder")
    print()
    
    # Load images
    lr_data, sr_data, k = load_images()
    
    if lr_data is None or sr_data is None:
        print("Please run the main.py script first to generate comparison images.")
        return
    
    # Create comparison view
    create_comparison_view(lr_data, sr_data, k)

if __name__ == "__main__":
    main()