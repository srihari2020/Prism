import streamlit as st
import subprocess
import os

st.set_page_config(page_title="Prism Generator", layout="centered")
st.title("Prism AI - Image Generator")
st.markdown("Test your fine-tuned `prism_target` LoRA weights here.")

prompt = st.text_area("Enter your Prompt:", "A cinematic portrait of prism_target in a cyberpunk neon city, highly detailed, 8k resolution, photorealistic")

if st.button("Generate Image", type="primary"):
    with st.spinner("Loading model and generating image... (~10 seconds)"):
        if os.path.exists("output.png"):
            os.remove("output.png") # Clear old image
            
        result = subprocess.run(["python", "generate_prism.py", prompt], capture_output=True, text=True)
        
        if os.path.exists("output.png"):
            st.image("output.png", caption=prompt, use_container_width=True)
            st.success("Generation complete! VRAM safely flushed.")
        else:
            st.error("Generation failed. Check terminal for details.")
            st.text(result.stderr[-2000:])
