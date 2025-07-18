# HiWave: High-Resolution Wavelet-based Image Generation

The minimal reimplement the core algorithm of paper *HiWave: Training-Free High-Resolution Image Generation via Wavelet-Based Diffusion Sampling*. Just for educational purposes (^.^)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run example
python main.py
```

## Usage

```python
from pipeline import CustomStableDiffusionPipeline

# Initialize pipeline
pipeline = CustomStableDiffusionPipeline()

# Generate image
image = pipeline.generate_image(
    prompt="A beautiful sunset over mountains",
    output_path="content/generated_image.png"
)

# Enhance existing image
enhanced = pipeline.hiwave(
    image=input_tensor,
    prompt="Enhanced version with better details"
)
```

## Results Comparison

### Full Image Comparison
| Initial Generation (1024x1024) | HiWave Enhanced (2048x2048) |
|-------------------|-----------------|
| ![Initial Generated](content/initial_generated_image.png) | ![High Resolution](content/high_resolution_landscape.png) |
| ![Initial Generated](content/initial_generated_image_1.png) | ![High Resolution](content/high_resolution_landscape_1.png) |

### Detail Comparison (Zoomed)
| Low-Resolution Detail | Super-Resolution Detail |
|----------------------|------------------------|
| ![LR Zoomed](content/lr_zoomed.png) | ![SR Zoomed](content/sr_zoomed.png) |
| ![LR Zoomed](content/lr_zoomed_1.png) | ![SR Zoomed](content/sr_zoomed_1.png) |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA GPU (recommended)

## Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate pywt pillow tqdm einops numpy matplotlib
```
