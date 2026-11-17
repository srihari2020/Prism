import torch
from unsloth import FastLanguageModel # Unsloth's dynamic loader
from peft import LoraConfig, get_peft_model

# We load the 4-bit quantized Qwen-Image model to respect the 8GB limit
model_name = "unsloth/Qwen-Image-2-7B-Instruct-GGUF"

print(f"Initializing Prism Training Pipeline...")
print(f"CUDA Available: {torch.cuda.is_available()}")

# Load model with dynamic 4-bit precision
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=True,
    device_map="sequential", # Forces efficient VRAM-to-RAM offloading
)

# Configure the Prism LoRA adapter
# Rank 8 is the sweet spot for identity/style capture on an 8GB GPU
prism_lora = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, prism_lora)
model.print_trainable_parameters()

# TODO: Tomorrow, we will map your dataset folder here and write the 
# Accelerate training loop using a batch size of 1.
