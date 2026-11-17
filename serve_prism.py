import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
from diffusers import StableDiffusionPipeline

# Global variable to hold our model pipeline
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("Loading Prism Inference Engine...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16,
        safety_checker=None # Fully uncensored
    )
    # Inject the trained LoRA adapter
    pipeline.unet.load_adapter("./models/prism_lora")
    pipeline.to("cuda")
    print("Prism Engine is live and ready for requests.")
    yield
    # Code here runs on shutdown
    print("Shutting down Prism Engine...")

app = FastAPI(title="Prism AI Engine", lifespan=lifespan)

class PromptRequest(BaseModel):
    prompt: str
    negative_prompt: str = "ugly, blurry, deformed, bad anatomy, watermark"

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

# This block actually starts the web server when you run the script
if __name__ == "__main__":
    uvicorn.run("serve_prism:app", host="0.0.0.0", port=8000, reload=True)
