from fastapi import FastAPI
from pydantic import BaseModel
import torch
from diffusers import DiffusionPipeline

app = FastAPI(title="Prism AI Engine")

class PromptRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""

@app.on_event("startup")
def load_prism_model():
    global pipeline
    # Load the base model and inject the Prism LoRA weights
    pipeline = DiffusionPipeline.from_pretrained(
        "unsloth/Qwen-Image-2-7B-Instruct-GGUF", 
        torch_dtype=torch.float16
    )
    # The safety checker is disabled to keep the model completely uncensored
    pipeline.safety_checker = None
    pipeline.load_lora_weights("./models/prism_lora")
    pipeline.to("cuda")
    print("Prism Engine is live.")

@app.post("/generate")
def generate_image(request: PromptRequest):
    # Generates the 1024x1024 base image for the Topaz upscaling pass
    image = pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        num_inference_steps=20,
        guidance_scale=7.5
    ).images[0]
    
    output_path = "output.png"
    image.save(output_path)
    return {"status": "success", "image_url": output_path}
