import torch
import os
from datasets import load_dataset
from unsloth import FastLanguageModel
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer

# 1. Initialize the 4-bit Quantized Model
model_name = "unsloth/Qwen-Image-2-7B-Instruct-GGUF"
print("Loading Prism Base Model in 4-bit...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=True,
    device_map="sequential",
)

# 2. Configure the LoRA Adapter (Rank 8 for 8GB VRAM)
prism_lora = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, prism_lora)

# 3. Load the Custom Dataset
# This reads the metadata.jsonl file we discussed and loads your cropped images
dataset = load_dataset(
    "json", 
    data_files={"train": "data/captions/metadata.jsonl"}, 
    split="train"
)

# 4. Define Memory-Efficient Training Arguments
training_args = TrainingArguments(
    output_dir="./models/prism_lora_checkpoints",
    per_device_train_batch_size=1,        # Crucial for 8GB VRAM
    gradient_accumulation_steps=4,        # Simulates a batch size of 4
    learning_rate=1e-4,
    logging_steps=10,
    max_steps=500,                        # Adjust based on dataset size (usually 100-500)
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",                   # 8-bit optimizer saves additional VRAM
    weight_decay=0.01,
    lr_scheduler_type="cosine",
)

# 5. Execute the Training Loop
print("Initializing Unsloth Gradient Descent...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=256,
    args=training_args,
)

trainer.train()

# 6. Save the Final Safetensors Adapter
print("Training Complete. Saving Prism LoRA weights...")
model.save_pretrained("./models/prism_lora")
print("Prism LoRA successfully exported to ./models/prism_lora")
