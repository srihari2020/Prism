import streamlit as st
import subprocess
import os

st.set_page_config(page_title="Prism Engine", layout="wide")
st.title("Prism AI - Local LoRA Studio")

tab1, tab2 = st.tabs(["Train Model", "Generate Image"])

with tab1:
    st.header("Fine-Tune LoRA")
    st.write("Ensure your images are in `data/raw_images/` and run `python fix_captions.py` if you added new ones.")
    
    if st.button("Start Training", type="primary"):
        with st.spinner("Training in progress... (Watch your Ubuntu terminal for the real-time loss curve)"):
            result = subprocess.run(["python", "train_prism.py"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Training Complete! LoRA saved to ./models/prism_lora")
            else:
                st.error("Training failed. Check terminal for details.")
                st.text(result.stderr[-2000:]) # Show last 2000 chars of error

with tab2:
    st.header("Test Generation")
    prompt = st.text_area("Enter your Prompt:", "A cinematic portrait of prism_target in a cyberpunk neon city, highly detailed, 8k resolution, photorealistic")
    
    if st.button("Generate Image", type="primary"):
        with st.spinner("Loading model and generating image... (~10 seconds)"):
            if os.path.exists("output.png"):
                os.remove("output.png") # Clear old image
                
            result = subprocess.run(["python", "generate_prism.py", prompt], capture_output=True, text=True)
            
            if os.path.exists("output.png"):
                st.image("output.png", caption=prompt, use_container_width=True)
                st.success("Generation complete! VRAM flushed.")
            else:
                st.error("Generation failed. Check terminal for details.")
                st.text(result.stderr[-2000:])
