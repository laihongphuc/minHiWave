import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL
import numpy as np
from typing import Optional, Tuple, List, Union
import pywt
from PIL import Image
import einops
import os
import random
import numpy as np
import time
from functools import wraps
from tqdm import tqdm

def profile_time(func_name: str = None):
    """
    Decorator to profile execution time of functions.
    
    Args:
        func_name: Optional custom name for the function in profiling output
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = func_name or func.__name__
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"[PROFILING] {name}: {execution_time:.2f} seconds")
            return result
        return wrapper
    return decorator


def set_seed(seed: int):
    """
    Set all random seeds for deterministic results.
    
    This function sets seeds for:
    - Python's random module
    - NumPy's random generator
    - PyTorch's CPU and CUDA random generators
    - CUDNN deterministic mode
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  


def get_time_ids(original_size, crops_coords_top_left, target_size, dtype=torch.float16):
    """
    original_size: (height, width) of original image
    crops_coords_top_left: (top, left) crop coordinates
    target_size: (height, width) of target output
    """
    time_ids = torch.tensor([
        original_size[0],           # original height
        original_size[1],           # original width  
        crops_coords_top_left[0],   # crop top
        crops_coords_top_left[1],   # crop left
        target_size[0],             # target height
        target_size[1]              # target width
    ], dtype=dtype)
    
    return time_ids.unsqueeze(0)  # Add batch dimension -> [1, 6]


def predicted_original_from_noise(latent, noise_pred, timestep, alphas_cumprod):
    alpha = alphas_cumprod[timestep]
    beta = 1 - alpha 
    predicted_original_sample = (latent - beta.sqrt() * noise_pred) / alpha.sqrt()
    return predicted_original_sample
    

def noise_from_predicted_original(latent, predicted_original, timestep, alphas_cumprod):
    alpha = alphas_cumprod[timestep]
    beta = 1 - alpha
    noise = (latent - alpha.sqrt() * predicted_original) / beta.sqrt()
    return noise

class CustomStableDiffusionPipeline:
    """
    Custom Stable Diffusion XL Pipeline with DDIM inversion and discrete wavelet sym4 methods.
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",  # SDXL default
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize the custom pipeline with individual components for SDXL.
        """
        self.model_id = model_id
        self.vae_id = "madebyollin/sdxl-vae-fp16-fix"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        
        # Load individual components
        self._load_components()
        
        # Initialize DDIM scheduler
        self.ddim_scheduler = DDIMScheduler.from_config(
            StableDiffusionXLPipeline.from_pretrained(model_id).scheduler.config
        )
        self.alphas_cumprod = self.ddim_scheduler.alphas_cumprod

        self.time_ids = get_time_ids((512, 512), (0, 0), (512, 512)).to(self.device)
        print(f"SDXL Pipeline initialized on device: {self.device}")
    
    def _load_components(self):
        """Load individual components from the SDXL pipeline."""
        print("Loading SDXL pipeline components...")
        # SDXL uses two text encoders and two tokenizers
        # 1. OpenCLIP (main)
        # First load the complete pipeline
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            # local_files_only=True
        )
        
        # Assign components from the pipeline
        self.tokenizer = pipeline.tokenizer
        self.text_encoder = pipeline.text_encoder.to(self.device, torch.float16)
        self.tokenizer_2 = pipeline.tokenizer_2  
        self.text_encoder_2 = pipeline.text_encoder_2.to(self.device, torch.float16)
        
        # Keep VAE loading the same
        self.vae = AutoencoderKL.from_pretrained(
            self.model_id,
            subfolder="vae", 
            torch_dtype=torch.float32,
            revision=None,
            variant=None,
            # local_files_only=True
        ).to(self.device)
        # self.vae = pipeline.vae.to(self.device, torch.float32)

        # Assign UNet from pipeline
        self.unet = pipeline.unet.to(self.device)
        
        # Clean up pipeline to free memory
        del pipeline
        print("All SDXL components loaded successfully!")

    def encode_prompt(self, prompt: str, negative_prompt: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text prompts to embeddings using both SDXL text encoders.
        Returns concatenated embeddings for both encoders.
        """
        # Tokenize prompts for both encoders
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        negative_text_inputs = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        negative_text_inputs_2 = self.tokenizer_2(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        # Encode to embeddings
        text_embed_list = []
        negative_embed_list = []
        with torch.no_grad():
            
            text_embeddings = self.text_encoder(
                text_inputs.input_ids.to(self.device),
                output_hidden_states=True,
            )
            text_embeddings_2 = self.text_encoder_2(
                text_inputs_2.input_ids.to(self.device),
                output_hidden_states=True,
            )

            text_embed_list = [
                text_embeddings.hidden_states[-2], 
                text_embeddings_2.hidden_states[-2], 
            ]

            negative_text_embeddings = self.text_encoder(
                negative_text_inputs.input_ids.to(self.device),
                output_hidden_states=True,
            )
            negative_text_embeddings_2 = self.text_encoder_2(
                negative_text_inputs_2.input_ids.to(self.device),
                output_hidden_states=True,
            )
            
            negative_embed_list = [
                negative_text_embeddings.hidden_states[-2], 
                negative_text_embeddings_2.hidden_states[-2], 
            ]

                
        return (torch.cat(text_embed_list, dim=-1), text_embeddings_2[0]), (torch.cat(negative_embed_list, dim=-1), negative_text_embeddings_2[0])

    def encode_image(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Encode image to latent space.
        
        Args:
            image: PIL Image or tensor
            
        Returns:
            Latent representation
        """
        if isinstance(image, Image.Image):
            # Convert PIL to tensor
            image = torch.from_numpy(np.array(image)).float() / 255.0
            image = image.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        image = image.to(self.device, dtype=torch.float32)
        
        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        return latents.to(self.torch_dtype)
    
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents back to image space.
        
        Args:
            latents: Latent representation
            
        Returns:
            Decoded image tensor
        """
        with torch.no_grad():
            latents = latents / self.vae.config.scaling_factor
            image = self.vae.decode(latents.to(torch.float32)).sample
        

        return image
    
    def ddim_inversion(
        self,
        image: Union[Image.Image, torch.Tensor],
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform DDIM inversion to find the initial noise.
        
        Args:
            image: Input image
            prompt: Text prompt
            num_inference_steps: Number of inversion steps
            guidance_scale: Guidance scale
            
        Returns:
            Initial noise tensor
        """
        print("Performing DDIM inversion...")
        
        # Encode image and prompt
        if latents is None:
            latents = self.encode_image(image)

        (text_embeddings, text_embeddings_2), (negative_text_embeddings, negative_text_embeddings_2) = self.encode_prompt(prompt)
        
        # Prepare scheduler
        self.ddim_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.ddim_scheduler.timesteps
        
        # Initialize noise
        noise = torch.randn_like(latents)
        
        # DDIM inversion loop
        for i, t in enumerate(reversed(timesteps)):
            # Prepare timestep
            timestep = torch.full((latents.shape[0],), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(
                    latents,
                    timestep,
                    encoder_hidden_states=text_embeddings,
                    added_cond_kwargs={
                        "text_embeds": text_embeddings_2,
                        "time_ids": self.time_ids
                    }
                ).sample
            
            # DDIM step
            latents = self.ddim_scheduler.step(
                noise_pred,
                t,
                latents,
                eta=0.0  # Deterministic
            ).prev_sample
        
        print("DDIM inversion completed!")
        return latents

    def discrete_wavelet_sym4(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply discrete wavelet transform using sym4 wavelet.
        
        Args:
            image: Input image tensor (B, C, H, W)
            level: Decomposition level (for dwt2, only 1 level is applied)
            
        Returns:
            Tuple of (cA, cH, cV, cD) tensors, each with shape (B, C, H//2, W//2)
        """
        batch_size, channels, height, width = image.shape
        
        # Initialize lists for each frequency component
        cA_list = []
        cH_list = []
        cV_list = []
        cD_list = []
        
        # Process each sample in the batch
        for b in range(batch_size):
            # Get single sample
            sample = image[b]  # (C, H, W)
            sample_np = sample.cpu().numpy()
            
            # Apply wavelet transform to each channel
            sample_cA = []
            sample_cH = []
            sample_cV = []
            sample_cD = []
            
            for c in range(channels):
                # Apply dwt2 to get 4 components: LL, LH, HL, HH
                coeffs = pywt.dwt2(sample_np[c], 'sym4')
                cA, (cH, cV, cD) = coeffs
                
                sample_cA.append(cA)
                sample_cH.append(cH)
                sample_cV.append(cV)
                sample_cD.append(cD)
            
            # Stack channels for each component
            cA_list.append(np.stack(sample_cA, axis=0))  # (C, H//2, W//2)
            cH_list.append(np.stack(sample_cH, axis=0))  # (C, H//2, W//2)
            cV_list.append(np.stack(sample_cV, axis=0))  # (C, H//2, W//2)
            cD_list.append(np.stack(sample_cD, axis=0))  # (C, H//2, W//2)
        
        # Stack batch results for each component
        cA_tensor = torch.from_numpy(np.stack(cA_list, axis=0)).float().to(self.device, dtype=image.dtype)
        cH_tensor = torch.from_numpy(np.stack(cH_list, axis=0)).float().to(self.device, dtype=image.dtype)
        cV_tensor = torch.from_numpy(np.stack(cV_list, axis=0)).float().to(self.device, dtype=image.dtype)
        cD_tensor = torch.from_numpy(np.stack(cD_list, axis=0)).float().to(self.device, dtype=image.dtype)
        
        return cA_tensor, cH_tensor, cV_tensor, cD_tensor

    def inverse_discrete_wavelet_sym4(self, cA: torch.Tensor, cH: torch.Tensor, cV: torch.Tensor, cD: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse discrete wavelet transform using sym4 wavelet.
        
        Args:
            cA: Approximation coefficients (B, C, H, W)
            cH: Horizontal detail coefficients (B, C, H, W)
            cV: Vertical detail coefficients (B, C, H, W)
            cD: Diagonal detail coefficients (B, C, H, W)
            
        Returns:
            Reconstructed image tensor (B, C, H*2, W*2)
        """
        batch_size, channels, height, width = cA.shape
        
        # Process each sample in the batch
        batch_results = []
        
        for b in range(batch_size):
            # Get single sample for each component
            sample_cA = cA[b].cpu().numpy()  # (C, H, W)
            sample_cH = cH[b].cpu().numpy()  # (C, H, W)
            sample_cV = cV[b].cpu().numpy()  # (C, H, W)
            sample_cD = cD[b].cpu().numpy()  # (C, H, W)
            
            # Reconstruct each channel
            reconstructed_channels = []
            
            for c in range(channels):
                # Reconstruct using inverse dwt2
                coeffs = (sample_cA[c], (sample_cH[c], sample_cV[c], sample_cD[c]))
                reconstructed = pywt.idwt2(coeffs, 'sym4')
                reconstructed_channels.append(reconstructed)
            
            # Stack channels for this sample
            sample_result = np.stack(reconstructed_channels, axis=0)  # (C, H*2, W*2)
            batch_results.append(sample_result)
        
        # Stack batch results
        result = np.stack(batch_results, axis=0)  # (B, C, H*2, W*2)
        result_tensor = torch.from_numpy(result).float().to(self.device, dtype=cA.dtype)
        
        return result_tensor
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        output_path: Optional[str] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate image using the custom pipeline.
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            width: Image width
            height: Image height
            seed: Random seed
            output_path: Path to save image
            
        Returns:
            Generated image tensor
        """
        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
            if self.device == "cuda":
                torch.cuda.manual_seed(seed)
        
        # Encode prompts
        (text_embeddings, text_embeddings_2), (negative_text_embeddings, negative_text_embeddings_2) = self.encode_prompt(prompt, negative_prompt)
        
        # Concatenate embeddings
        encoder_hidden_states = torch.cat([negative_text_embeddings, text_embeddings], dim=0)
        pool_hidden_states = torch.cat([negative_text_embeddings_2, text_embeddings_2], dim=0)
        # Prepare scheduler
        self.ddim_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.ddim_scheduler.timesteps
        
        # Initialize latents
        latents = torch.randn(
            (1, self.unet.config.in_channels, height // 8, width // 8),
            device=self.device,
            dtype=self.torch_dtype
        ) if latents is None else latents 
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.ddim_scheduler.scale_model_input(latent_model_input, t)
            

            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs={
                        "text_embeds": pool_hidden_states,
                        "time_ids": torch.cat([self.time_ids, self.time_ids], dim=0),
                    }
                ).sample
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous sample
            latents = self.ddim_scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode latents
        image = self.decode_latents(latents)
        
        # Save if path provided
        if output_path:
            self._save_image(image, output_path)
        
        return image
    
    def _save_image(self, image_tensor: torch.Tensor, output_path: str):
        """Save image tensor to file."""
        # Convert to PIL and save
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image_np = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        image_np = np.transpose(image_np, (1, 2, 0))
        pil_image = Image.fromarray(image_np)
        
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        pil_image.save(output_path)
        print(f"Image saved to: {output_path}")
    
    def apply_wavelet_denoising(
        self,
        image: torch.Tensor,
        level: int = 3,
        threshold: float = 0.1
    ) -> torch.Tensor:
        """
        Apply wavelet-based denoising using sym4 wavelet.
        
        Args:
            image: Input image tensor
            level: Wavelet decomposition level
            threshold: Threshold for coefficient filtering
            
        Returns:
            Denoised image tensor
        """
        # Apply wavelet transform
        approximations, details = self.discrete_wavelet_sym4(image, level)
        
        # Apply thresholding to details
        thresholded_details = []
        for detail_level in details:
            thresholded_level = []
            for detail in detail_level:
                # Soft thresholding
                thresholded = torch.sign(torch.from_numpy(detail)) * torch.clamp(
                    torch.abs(torch.from_numpy(detail)) - threshold, min=0
                ).numpy()
                thresholded_level.append(thresholded)
            thresholded_details.append(thresholded_level)
        
        # Inverse transform
        denoised_image = self.inverse_discrete_wavelet_sym4(approximations, thresholded_details)
        
        return denoised_image
    
    @torch.no_grad()
    @profile_time("HiWave Total")
    def hiwave(
            self, 
            image: torch.Tensor, 
            prompt: str,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            seed: Optional[int] = None,
            show_progress: bool = True,
        ):
        """
        Produce high-resolution image from low-resolution image
        """
        # Set seed for deterministic results
        if seed is not None:
            set_seed(seed)
            
        init_high_resolution = F.interpolate(image, scale_factor=2, mode='bilinear')
        latent_high_resolution = self.encode_image(init_high_resolution)
        anchor_mean, anchor_std = latent_high_resolution.mean(), latent_high_resolution.std()
        views = self.get_views(height=latent_high_resolution.shape[2], width=latent_high_resolution.shape[3])
        patch_wise = []
        for h_start, h_end, w_start, w_end in views:
            patch_wise.append(latent_high_resolution[:, :, h_start:h_end, w_start:w_end])
        patch_wise = torch.cat(patch_wise, dim=0)

        add_time_ids_input = []
        for h_start, h_end, w_start, w_end in views:
            add_time_ids = self.time_ids
            add_time_ids[:, 2] = h_start * 8
            add_time_ids[:, 3] = w_start * 8
            add_time_ids_input.append(add_time_ids)
        add_time_ids_input = torch.cat(add_time_ids_input, dim=0)

        # patch_wise = einops.rearrange(patch_wise, "b c (c1 h) (c2 w) -> (b c1 c2) c h w", c1=2, c2=2)

        (text_embeddings, text_embeddings_2), (negative_text_embeddings, negative_text_embeddings_2) = self.encode_prompt(prompt)
        # Concatenate embeddings
        encoder_hidden_states = torch.cat([negative_text_embeddings, text_embeddings])
        pool_hidden_states = torch.cat([negative_text_embeddings_2, text_embeddings_2])
        
        # Prepare scheduler
        self.ddim_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.ddim_scheduler.timesteps
        
        # Initialize noise
        noise = torch.randn_like(patch_wise)
        
        ddim_inverted_patch_wise = []
        repeat_times = patch_wise.shape[0]
        # DDIM inversion loop
        text_embeddings_ddim_inv = einops.repeat(text_embeddings, "b seq d -> (repeat b) seq d", repeat=repeat_times)
        text_embeddings_2_ddim_inv = einops.repeat(text_embeddings_2, "b d -> (repeat b) d", repeat=repeat_times)
        # time_ids_ddim_inv = einops.repeat(self.time_ids, "b d -> (repeat b) d", repeat=repeat_times)
        
        # Progress bar for DDIM inversion
        if show_progress:
            print("Starting DDIM inversion...")
            pbar_inversion = tqdm(reversed(timesteps), total=len(timesteps), desc="DDIM Inversion")
        else:
            pbar_inversion = reversed(timesteps)
            
        for i, t in enumerate(pbar_inversion):
            # Prepare timestep
            timestep = torch.full((repeat_times,), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(
                    patch_wise,
                    timestep,
                    encoder_hidden_states=text_embeddings_ddim_inv,
                    added_cond_kwargs={
                        "text_embeds": text_embeddings_2_ddim_inv,
                        "time_ids": add_time_ids_input
                    }
                ).sample
            
            current_t = max(0, t.item() - (1000 // num_inference_steps))
            next_t = t
            alpha_t = self.alphas_cumprod[current_t]
            alpha_t_next = self.alphas_cumprod[next_t]
                    # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
            patch_wise = (patch_wise - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
                1 - alpha_t_next
            ).sqrt() * noise_pred

            ddim_inverted_patch_wise.append(patch_wise)
        

        # DDIM loop 
        encoder_hidden_states_ddim_loop = einops.repeat(encoder_hidden_states, "b seq d -> (b repeat) seq d", repeat=repeat_times)
        pool_hidden_states_ddim_loop = einops.repeat(pool_hidden_states, "b d -> (b repeat) d", repeat=repeat_times)
        # time_ids_ddim_loop = einops.repeat(self.time_ids, "b d -> (b repeat) d", repeat=repeat_times * 2)
        
        # Progress bar for DDIM loop
        if show_progress:
            print("Starting DDIM loop...")
            pbar_loop = tqdm(timesteps, total=len(timesteps), desc="DDIM Loop")
        else:
            pbar_loop = timesteps
            
        for i, t in enumerate(pbar_loop):
            cosine_factor = 0.5 * (1 + torch.cos(torch.pi * (self.ddim_scheduler.config.num_train_timesteps  - t) / self.ddim_scheduler.config.num_train_timesteps )).cpu()
            c1 = cosine_factor ** 2
            if i <= 15:
                patch_wise = patch_wise * (c1) + ddim_inverted_patch_wise[-(i)] * (1 - c1)
            # DemoDiffusion
            count = torch.zeros_like(latent_high_resolution)
            value = torch.zeros_like(latent_high_resolution)
            # Prepare timestep
            timestep = torch.full((repeat_times,), t, device=self.device, dtype=torch.long)

            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([patch_wise] * 2)
            latent_model_input = self.ddim_scheduler.scale_model_input(latent_model_input, t)
            
            add_time_ids_input = []
            for h_start, h_end, w_start, w_end in views:
                add_time_ids = torch.cat([self.time_ids, self.time_ids]).clone()
                add_time_ids[:, 2] = h_start * 8
                add_time_ids[:, 3] = w_start * 8
                add_time_ids_input.append(add_time_ids)
            add_time_ids_input = torch.cat(add_time_ids_input, dim=0)
            add_time_ids_input = einops.rearrange(add_time_ids_input, "(b c1) d -> (c1 b) d", c1=2)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states_ddim_loop,
                    added_cond_kwargs={
                        "text_embeds": pool_hidden_states_ddim_loop,
                        "time_ids": add_time_ids_input,
                    }
                ).sample
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            predicted_original_text = predicted_original_from_noise(patch_wise, noise_pred_text, t, self.alphas_cumprod)
            predicted_original_uncond = predicted_original_from_noise(patch_wise, noise_pred_uncond, t, self.alphas_cumprod)
            cA_text, cH_text, cV_text, cD_text = self.discrete_wavelet_sym4(predicted_original_text)
            cA_uncond, cH_uncond, cV_uncond, cD_uncond = self.discrete_wavelet_sym4(predicted_original_uncond)
            cH_text += (guidance_scale - 1) * (cH_text - cH_uncond)
            cV_text += (guidance_scale - 1) * (cV_text - cV_uncond)
            cD_text += (guidance_scale - 1) * (cD_text - cD_uncond)
            predicted_original = self.inverse_discrete_wavelet_sym4(cA_text, cH_text, cV_text, cD_text)
            noise_pred = noise_from_predicted_original(patch_wise, predicted_original, t, self.alphas_cumprod)
            
            # Compute previous sample
            patch_wise = self.ddim_scheduler.step(noise_pred, t, patch_wise).prev_sample

            for patch, (h_start, h_end, w_start, w_end) in zip(patch_wise, views):
                count[:, :, h_start:h_end, w_start:w_end] += 1
                value[:, :, h_start:h_end, w_start:w_end] += patch

            # patch_wise = torch.where(count > 0, value / count, value)
            latent_high_resolution = torch.where(count > 0, value / count, value)

            patch_wise = []
            for h_start, h_end, w_start, w_end in views:
                patch_wise.append(latent_high_resolution[:, :, h_start:h_end, w_start:w_end])
            patch_wise = torch.cat(patch_wise, dim=0)

        latent_high_resolution = (latent_high_resolution - latent_high_resolution.mean()) / latent_high_resolution.std()
        latent_high_resolution = latent_high_resolution * anchor_std + anchor_mean

        # latent_high_resolution = einops.rearrange(patch_wise, "(b c1 c2) c h w -> b c (c1 h) (c2 w)", c1=2, c2=2)
        image_high_resolution = self.decode_latents(latent_high_resolution)

        return image_high_resolution
    
    def get_profiling_info(self) -> dict:
        """
        Get profiling information for the last hiwave run.
        
        Returns:
            Dictionary containing profiling data
        """
        # This can be extended to store and return detailed profiling info
        return {
            "method": "hiwave",
            "components": ["DDIM Inversion", "DDIM Loop", "Wavelet Processing"],
            "note": "Use @profile_time decorator for detailed timing"
        }
    
    def get_views(
        self,
        height: int,
        width: int, 
        stride: int = 64,
        window_size: int = 128,
        random_jitter: bool = False,
    ):
        num_blocks_height = int((height - window_size) / stride - 1e-6) + 2 if height > window_size else 1
        num_blocks_width = int((width - window_size) / stride - 1e-6) + 2 if width > window_size else 1
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        views = []
        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride)
            h_end = h_start + window_size
            w_start = int((i % num_blocks_width) * stride)
            w_end = w_start + window_size

            if h_end > height:
                h_start = int(h_start + height - h_end)
                h_end = int(height)
            if w_end > width:
                w_start = int(w_start + width - w_end)
                w_end = int(width)
            if h_start < 0:
                h_end = int(h_end - h_start)
                h_start = 0
            if w_start < 0:
                w_end = int(w_end - w_start)
                w_start = 0

            if random_jitter:
                jitter_range = (window_size - stride) // 4
                w_jitter = 0
                h_jitter = 0
                if (w_start != 0) and (w_end != width):
                    w_jitter = random.randint(-jitter_range, jitter_range)
                elif (w_start == 0) and (w_end != width):
                    w_jitter = random.randint(-jitter_range, 0)
                elif (w_start != 0) and (w_end == width):
                    w_jitter = random.randint(0, jitter_range)
                if (h_start != 0) and (h_end != height):
                    h_jitter = random.randint(-jitter_range, jitter_range)
                elif (h_start == 0) and (h_end != height):
                    h_jitter = random.randint(-jitter_range, 0)
                elif (h_start != 0) and (h_end == height):
                    h_jitter = random.randint(0, jitter_range)
                h_start += (h_jitter + jitter_range)
                h_end += (h_jitter + jitter_range)
                w_start += (w_jitter + jitter_range)
                w_end += (w_jitter + jitter_range)
            
            views.append((h_start, h_end, w_start, w_end))
        return views

