import subprocess
import os

print("Starting Prism Engine: Top-Tier SDXL LoRA Fine-Tuning...")

# Keep fragmentation management active to ensure the 8GB VRAM stays clean
env = os.environ.copy()
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

command = [
    "python", "sdxl_trainer.py",
    "--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
    "--pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix", # SDXL VAE fix to prevent NaN errors in fp16
    "--instance_data_dir=data/raw_images",
    "--instance_prompt=A cinematic portrait of prism_target",
    "--output_dir=models/prism_sdxl_lora",
    "--mixed_precision=fp16",
    "--train_batch_size=1",
    "--gradient_accumulation_steps=4",
    "--learning_rate=1e-4",
    "--max_train_steps=500",
    "--gradient_checkpointing",
    "--use_8bit_adam",
    "--resolution=1024",  # SDXL natively trains at stunning 1024x1024 resolution
    "--seed=42"
]

subprocess.run(command, env=env)
