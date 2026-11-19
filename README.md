python3 - << 'PYEOF'
from huggingface_hub import HfApi

readme = """---
license: mit
pipeline_tag: text-to-image
library_name: diffusers
base_model:
  - SG161222/RealVisXL_V5.0_Lightning
  - cagliostrolab/animagine-xl-3.1
tags:
  - stable-diffusion-xl
  - sdxl
  - diffusers
  - text-to-image
  - photorealism
  - anime
  - streamlit
  - local-ai
---

# Prism Auto Studio

**Prism Auto Studio** is a zero-config, self-contained local image generation studio built with **Diffusers** and **Streamlit**. It automatically routes prompts between photorealism and 2D anime, applies Danbooru character tagging, selects optimal aspect ratios, and runs an always-on 1.5x Hires detail pass—optimized to run smoothly on **8GB VRAM GPUs**.

## Key Features
- **Automatic Dual-Engine Routing**: Detects whether your prompt calls for photorealism or 2D anime and loads the matching engine automatically.
  - **Photorealism Engine (`models/realvisxl`)**: Powered by **RealVisXL V5.0 Lightning** (10 steps, CFG `2.2`, SDE DPM-Solver++ Karras).
  - **Anime Engine (`models/animagine`)**: Powered by **Animagine XL 3.1** (22 steps, CFG `6.5`, Euler Ancestral).
- **Smart Prompt & Character Tagging**: Strips conversational filler phrases and maps popular anime characters to their exact Danbooru tags for accurate likeness.
- **Always-On 1.5x Hires Fix**: Runs a second-stage `img2img` detail pass (`strength=0.28`) to sharpen faces, hands, and textures.
- **8GB VRAM Optimized**: Uses `bfloat16`, `enable_model_cpu_offload()`, VAE tiling, and automatic CUDA cache clearing when switching engines.
- **Zero-Config UI**: Minimal interface—type a prompt, hit Generate, and download the output.

## Repository Structure
```text
prism-auto-studio/
├── app.py                  # Streamlit application & auto-routing logic
├── requirements.txt        # Python dependencies
├── README.md               # Model card & documentation
└── models/
    ├── realvisxl/          # RealVisXL V5.0 Lightning (diffusers weights)
    └── animagine/          # Animagine XL 3.1 (diffusers weights)
