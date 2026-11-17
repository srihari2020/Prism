import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torchvision import transforms
import bitsandbytes as bnb

# 1. Load the Base Diffusion Model
# We are using SD 1.5 as the base here because it safely fits in 8GB VRAM for training.
# Once training works, you can swap this path to an uncensored local model from Civitai.
model_id = "runwayml/stable-diffusion-v1-5"
print(f"Loading Prism Base Model: {model_id}")

pipeline = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    safety_checker=None # Disabled for uncensored generation
)
pipeline.to("cuda")

unet = pipeline.unet
text_encoder = pipeline.text_encoder
vae = pipeline.vae
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# Freeze VAE and Text Encoder to save massive amounts of VRAM
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.train()

# Enable gradient checkpointing to fit training inside 8GB VRAM
unet.enable_gradient_checkpointing()

# 2. Configure the LoRA Adapter
prism_lora = LoraConfig(
    r=8,
    lora_alpha=16,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"], # Standard attention layers
)
unet = get_peft_model(unet, prism_lora)
unet.print_trainable_parameters()

# 3. Memory-Efficient 8-bit Optimizer
optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=1e-4, weight_decay=0.01)

# 4. Dataset Loading and Preprocessing
# Assumes images are in data/raw_images and metadata in data/captions/metadata.jsonl
try:
    dataset = load_dataset("json", data_files={"train": "data/captions/metadata.jsonl"}, split="train")
except Exception as e:
    print("Dataset not found. Please ensure data/captions/metadata.jsonl is created.")
    exit()

image_transforms = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def tokenize_captions(examples):
    inputs = pipeline.tokenizer(
        examples["text"], max_length=pipeline.tokenizer.model_max_length, 
        padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

# 5. The Core Training Loop
epochs = 10
print("Initializing Prism Gradient Descent...")

for epoch in range(epochs):
    epoch_loss = 0
    for step, batch in enumerate(dataset):
        # Load and transform image
        from PIL import Image
        image = Image.open(batch["file_name"]).convert("RGB")
        pixel_values = image_transforms(image).unsqueeze(0).to("cuda", dtype=torch.float16)
        
        # Tokenize text
        input_ids = pipeline.tokenizer(
            batch["text"], padding="do_not_pad", truncation=True, return_tensors="pt"
        ).input_ids.to("cuda")
        
        # Convert images to latent space
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        
        # Sample noise and add to latents
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device="cuda")
        timesteps = timesteps.long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get text embeddings
        encoder_hidden_states = text_encoder(input_ids)[0]
        
        # Predict the noise residual
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        
        # Calculate loss and backpropagate
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss / len(dataset)}")

# 6. Save the Final Adapter
print("Training Complete. Saving Prism LoRA weights...")
unet.save_pretrained("./models/prism_lora")
print("Prism LoRA successfully exported to ./models/prism_lora")
