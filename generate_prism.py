import torch, sys
from diffusers import StableDiffusionPipeline

# Accept prompt from CLI argument or use default
prompt = sys.argv[1] if len(sys.argv) > 1 else "A cinematic portrait of prism_target in a cyberpunk neon city, highly detailed, 8k resolution, photorealistic"

print("Loading Base Model into VRAM...")
pipe = StableDiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE", 
    torch_dtype=torch.float16
).to("cuda")

print("Injecting your custom Prism LoRA...")
pipe.load_lora_weights("./models/prism_lora")

print(f"Generating: {prompt}")
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("output.png")
print("Success! Image saved as output.png")
