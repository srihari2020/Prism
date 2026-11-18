import sys, torch
from diffusers import FluxPipeline

prompt = sys.argv[1] if len(sys.argv) > 1 else "A cinematic portrait of a person in a cyberpunk neon city, highly detailed, 8k resolution"

print("Loading FLUX into VRAM (with 8GB optimization)...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", 
    torch_dtype=torch.bfloat16
)

# The magic line that makes FLUX fit on your RTX 5060
pipe.enable_model_cpu_offload()

# Note: Your SD1.5 LoRA is commented out until we train a new FLUX one
# pipe.load_lora_weights("./models/prism_lora")

print(f"Generating with FLUX: {prompt}")
# FLUX Schnell is incredibly fast and only requires 4 steps with 0 guidance
image = pipe(
    prompt, 
    num_inference_steps=4, 
    guidance_scale=0.0
).images[0]

image.save("output.png")
print("Success! Image saved to output.png.")
