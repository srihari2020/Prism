import os
import gc
import time
import random
import torch
import streamlit as st
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
)

st.set_page_config(page_title="Prism", layout="centered", initial_sidebar_state="collapsed")
st.markdown(
    "<style>#MainMenu, header, footer {visibility: hidden;}</style>",
    unsafe_allow_html=True,
)
os.makedirs("outputs", exist_ok=True)

ANIME_KEYWORDS = [
    "anime", "manga", "2d", "waifu", "husbando", "chibi", "cel shaded", "ghibli",
    "gojo", "satoru", "sukuna", "yuji", "megumi", "naruto", "sasuke", "goku",
    "vegeta", "luffy", "zoro", "ichigo", "jinwoo", "sung jin", "saitama", "levi",
    "mikasa", "eren", "makima", "denji", "tanjiro", "nezuko", "zero two", "rem",
    "hatsune miku", "jjk", "jujutsu", "solo leveling", "demon slayer", "one piece",
    "dragon ball", "bleach", "attack on titan", "chainsaw man", "hentai", "ecchi",
]

import re

CHAR_TAGS = {
    "gojo": "1boy, solo, gojou satoru, jujutsu kaisen, white hair, spiky hair, blue eyes, black blindfold, dark blue high-collar uniform",
    "sukuna": "1boy, solo, ryoumen sukuna, jujutsu kaisen, pink hair, facial markings, kimono",
    "yuji": "1boy, solo, itadori yuuji, jujutsu kaisen, pink hair, red hoodie, dark blue uniform",
    "naruto": "1boy, solo, uzumaki naruto, naruto, blonde hair, whisker markings, forehead protector, orange jacket",
    "sasuke": "1boy, solo, uchiha sasuke, naruto, black hair, sharingan, dark blue outfit",
    "goku": "1boy, solo, son goku, dragon ball, spiky black hair, orange gi",
    "luffy": "1boy, solo, monkey d. luffy, one piece, straw hat, scar under eye, red vest",
    "zoro": "1boy, solo, roronoa zoro, one piece, green hair, three swords, green haramaki",
    "jinwoo": "1boy, solo, sung jin-woo, solo leveling, black hair, glowing blue eyes, dark coat",
    "levi": "1boy, solo, levi ackerman, shingeki no kyojin, black undercut hair, survey corps uniform, green cloak",
    "makima": "1girl, solo, makima, chainsaw man, red braided hair, yellow ringed eyes, white shirt, black tie",
    "tanjiro": "1boy, solo, kamado tanjirou, kimetsu no yaiba, dark red hair, forehead scar, green and black checkered haori",
}

def analyze_prompt(user_prompt: str):
    cleaned = re.sub(r"^(generate|create|make|draw|show me)\s+(a|an)?\s*(image|picture|photo|pic|art|drawing)?\s*(of)?\s*", "", user_prompt.strip(), flags=re.IGNORECASE)
    p = cleaned.lower()
    is_anime = any(kw in p for kw in ANIME_KEYWORDS)

    for key, tags in CHAR_TAGS.items():
        if key in p:
            cleaned = f"{tags}, {cleaned}"
            is_anime = True
            break
    user_prompt = cleaned

    if any(w in p for w in ["wide", "widescreen", "cinematic", "landscape", "wallpaper", "scenery", "city"]):
        width, height = (1216, 832)
    elif any(w in p for w in ["portrait", "vertical", "phone", "selfie", "close-up", "closeup", "full body", "standing"]):
        width, height = (832, 1216)
    else:
        width, height = (1024, 1024)

    do_hires = True

    if is_anime:
        return {
            "mode": "anime",
            "width": width,
            "height": height,
            "do_hires": do_hires,
            "prompt": f"masterpiece, best quality, very aesthetic, absurdres, {user_prompt}",
            "neg_prompt": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, blurry, 3d, realistic, photo",
            "steps": 22,
            "cfg": 6.5,
        }
    return {
        "mode": "realism",
        "width": width,
        "height": height,
        "do_hires": do_hires,
        "prompt": f"{user_prompt}, ultra detailed, natural lighting, sharp focus, raw candid photo",
        "neg_prompt": "worst quality, low quality, blurry, deformed, bad anatomy, extra fingers, mutated hands, plastic skin, airbrushed, watermark, text, cgi, 3d render, cartoon",
        "steps": 10,
        "cfg": 2.2,
    }

@st.cache_resource(max_entries=1, show_spinner="Loading engine...")
def load_engine(mode: str):
    gc.collect()
    torch.cuda.empty_cache()

    model_id = "cagliostrolab/animagine-xl-3.1" if mode == "anime" else "SG161222/RealVisXL_V5.0_Lightning"
    txt2img = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )

    if mode == "anime":
        txt2img.scheduler = EulerAncestralDiscreteScheduler.from_config(txt2img.scheduler.config)
    else:
        txt2img.scheduler = DPMSolverMultistepScheduler.from_config(
            txt2img.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )

    txt2img.enable_model_cpu_offload()
    txt2img.vae.enable_tiling()

    img2img = StableDiffusionXLImg2ImgPipeline(**txt2img.components)
    img2img.enable_model_cpu_offload()
    img2img.vae.enable_tiling()
    return txt2img, img2img

st.title("Prism")

user_prompt = st.text_area(
    "Prompt",
    placeholder="Describe anything...",
    height=100,
    label_visibility="collapsed",
)

if st.button("Generate", type="primary", use_container_width=True) and user_prompt.strip():
    plan = analyze_prompt(user_prompt.strip())
    txt2img, img2img = load_engine(plan["mode"])
    seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator("cuda").manual_seed(seed)

    with st.spinner("Generating..."):
        image = txt2img(
            prompt=plan["prompt"],
            negative_prompt=plan["neg_prompt"],
            num_inference_steps=plan["steps"],
            guidance_scale=plan["cfg"],
            width=plan["width"],
            height=plan["height"],
            generator=generator,
        ).images[0]
        torch.cuda.empty_cache()

        if plan["do_hires"]:
            new_w = int((plan["width"] * 1.5) // 64 * 64)
            new_h = int((plan["height"] * 1.5) // 64 * 64)
            upscaled = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            strength = 0.28
            image = img2img(
                prompt=plan["prompt"],
                negative_prompt=plan["neg_prompt"],
                image=upscaled,
                strength=strength,
                num_inference_steps=max(6, int(plan["steps"] / strength)),
                guidance_scale=plan["cfg"],
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
            gc.collect()
            torch.cuda.empty_cache()

    out_path = f"outputs/prism_{int(time.time())}_{seed}.png"
    image.save(out_path)

    st.image(image, use_container_width=True)
    with open(out_path, "rb") as f:
        st.download_button(
            label="Download",
            data=f,
            file_name=os.path.basename(out_path),
            mime="image/png",
            use_container_width=True,
        )
