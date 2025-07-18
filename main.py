"""
HiWave: High-Resolution Wavelet-based Image Generation
Main script for using the HiWave pipeline.
"""

from pipeline import CustomStableDiffusionPipeline
from PIL import Image
import torch
import numpy as np


def main():
    """Example usage of HiWave pipeline."""
    # Initialize pipeline
    pipeline = CustomStableDiffusionPipeline()
    
    # Example 1: Generate a new image
    print("Generating a new image...")
    image = pipeline.generate_image(
        prompt="A beautiful sunset over mountains, high resolution, detailed",
        negative_prompt="blurry, low quality, distorted",
        num_inference_steps=30,
        guidance_scale=7.5,
        width=1024,
        height=1024,
        seed=42,
        output_path="content/generated_image.png"
    )
    
    # Example 2: Enhance an existing image
    print("Enhancing an existing image...")
    # Load an image (you would replace this with your actual image path)
    input_image = Image.open("content/generated_image.png")
    input_tensor = torch.from_numpy(np.array(input_image)).float() / 255.0
    input_tensor = input_tensor.unsqueeze(0).permute(0, 3, 1, 2)
    
    enhanced_image = pipeline.hiwave(
        image=input_tensor,
        prompt="Enhanced version with better details and clarity",
        num_inference_steps=30,
        guidance_scale=5.0,
        seed=42
    )
    
    pipeline._save_image(enhanced_image, "content/enhanced_image.png")
    
    print("Done! Check the content folder for generated images.")


if __name__ == "__main__":
    main()


