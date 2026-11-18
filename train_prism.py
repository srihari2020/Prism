import subprocess

print("Starting Prism Engine: FLUX LoRA Fine-Tuning...")

command = [
    "accelerate", "launch", "flux_trainer.py",
    "--pretrained_model_name_or_path=black-forest-labs/FLUX.1-schnell",
    "--instance_data_dir=data/raw_images",
    "--instance_prompt=A cinematic portrait of prism_target",
    "--output_dir=models/prism_flux_lora",
    "--mixed_precision=bf16",
    "--train_batch_size=1",
    "--gradient_accumulation_steps=4",
    "--learning_rate=1e-4",
    "--max_train_steps=500",
    "--gradient_checkpointing",
    "--use_8bit_adam",
    "--resolution=512",
    "--seed=42"
]

subprocess.run(command)
