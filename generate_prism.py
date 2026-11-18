import sys, torch
from diffusers import FluxPipeline, FluxTransformer2DModel

prompt = sys.argv[1] if len(sys.argv) > 1 else "A cinematic portrait of prism_target in a cyberpunk neon city, highly detailed, 8k resolution, photorealistic"

print("Loading ultra-light GGUF core into Prism Engine...")

# Diffusers 0.44.0.dev0 natively supports loading GGUF weights!
transformer = FluxTransformer2DModel.from_single_file(
    "models/gguf/flux1-schnell-q4.gguf",
    torch_dtype=torch.bfloat16
)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", 
    transformer=transformer,
    torch_dtype=torch.bfloat16
)

pipe.enable_model_cpu_offload()

# Note: Your LoRA loading will go here once the training is complete
# pipe.load_lora_weights("./models/prism_flux_lora")

print(f"Generating Fast GGUF Image: {prompt}")
image = pipe(
    prompt, 
    num_inference_steps=4, 
    guidance_scale=0.0
).images[0]

image.save("output.png")
print("Success! GGUF generation complete.")
