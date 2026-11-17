from fastapi import FastAPI
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionPipeline

app = FastAPI(title="Prism AI Engine")

class PromptRequest(BaseModel):
    prompt: str
    negative_prompt: str = "ugly, blurry, deformed, bad anatomy, watermark"

@app.on_event("startup")
def load_prism_model():
    global pipeline
    print("Loading Prism Inference Engine...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16,
        safety_checker=None # Fully uncensored
    )
    # Inject the trained LoRA
    pipeline.load_lora_weights("./models/prism_lora")
    pipeline.to("cuda")
    print("Prism Engine is live.")

@app.post("/generate")
def generate_image(request: PromptRequest):
    print(f"Generating: {request.prompt}")
    image = pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]
    
    output_path = "output.png"
    image.save(output_path)
    return {"status": "success", "image_url": output_path}
