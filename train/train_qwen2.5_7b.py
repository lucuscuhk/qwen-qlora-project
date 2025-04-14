"""
train_qwen2.5-7b.py

This script fine-tunes the Qwen2.5-7B-Instruct model using QLoRA with 4-bit NF4 quantization.
It leverages Hugging Face Accelerate for multi-GPU training, integrates Weights & Biases (wandb) for experiment tracking,
and saves the fine-tuned LoRA adapter weights (not the full base model) and tokenizer for efficient storage and reuse.

Main Features:
- Efficient 4-bit training via BitsAndBytes (NF4 quantization + double quant)
- LoRA adapters inserted into transformer attention and MLP modules
- Tokenization with max_length=512 and label masking for supervised fine-tuning
- Full Accelerate-compatible training loop with gradient accumulation
- Real-time loss and learning rate logging via wandb

🔧 Training Configuration:
- Model: Qwen2.5-7B-Instruct
- Tokenizer: Loaded from same directory (trust_remote_code=True)
- Batch size: 5
- Gradient accumulation steps: 4
- Effective batch size: 5 x 4 = 20
- Learning rate: 2e-4
- Epochs: 3
- Max sequence length: 512 tokens
- Optimizer: AdamW
- LR scheduler: CosineAnnealingLR

🧩 LoRA Configuration:
- Rank (r): 64
- Alpha: 32
- Dropout: 0.05
- Target modules:
  ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

📊 Dataset:
- Format: JSONL with fields ["instruction", "input", "output"]
- Source: ../data/100k_qllora_format.jsonl (dataset: https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k)
- Preprocessed into token IDs with masked labels for supervised learning

💾 Output:
- Saves LoRA adapter weights via `save_pretrained()`
- Saves tokenizer for downstream inference
- Output directory: ./adapter_qwen2.5-7b

🧠 Memory Usage (on 8x NVIDIA RTX 4090):
- Observed memory: ~22286MiB / 24564MiB per GPU
- Training fits comfortably with batch_size=5 and gradient_accumulation_steps=4

Author: Lucus
Date: 2025-04-13
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from peft.utils.other import prepare_model_for_kbit_training
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import default_data_collator
import wandb
from tqdm.auto import tqdm
import time

# initialize the wandb (only in the main process)
if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    wandb.init(
        project="qwen-qlora-finance",
        name="qwen2-7b-qlora-accelerate",
        config={"lr": 2e-4, "batch_size": 5, "epochs": 3, "model": "Qwen2.5-7B-Instruct"}
    )

# Accelerator initialization

accelerator = Accelerator()
device = accelerator.device

# Convert to an absolute path to prevent it from being mistaken for repo_id (avoid starting from ".")
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "..", "Qwen2.5-7B-Instruct")
model_dir = os.path.abspath(model_dir)  

data_path = "../data/100k_qllora_format.jsonl"
output_dir = "../adapter&tokenizer_qwen2.5-7b"

# BitsAndBytes config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# load the model
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config,
    trust_remote_code=True,
    local_files_only=True,
    device_map=None
)
# Tell the model which token is used for padding
model.config.pad_token_id = tokenizer.pad_token_id

# insert LoRA
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    # trainable adapter for qwen2.5-7b
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load a Dataset object from a local JSONL file for training
raw_dataset = load_dataset("json", data_files=data_path, split="train")

def format(example):
    instruction, input_text, output = example["instruction"], example["input"], example["output"]
    if input_text:
        prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    full_text = prompt + output + "<|im_end|>"
    tokenized = tokenizer(full_text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
    input_ids = tokenized["input_ids"][0]
    prompt_len = len(tokenizer(prompt, truncation=True, max_length=512)["input_ids"])
    labels = input_ids.clone()
    labels[:prompt_len] = -100 # avoid letting the model learn user prompt
    # HuggingFace dataset return python list in default
    return {"input_ids": input_ids.tolist(), "labels": labels.tolist()} 


# remove original columns to avoid keeping unused text fields and prevent Trainer errors
tokenized_dataset = raw_dataset.map(format, remove_columns=["instruction", "input", "output"])

# data loader: transfer raw_data into trainable tensor
train_dataloader = DataLoader(
    tokenized_dataset,
    batch_size=5,  # experiment: 8 * 4090, memory usage: 22286MiB/ 24564MiB
    shuffle=True,
    # see more data collector: https://huggingface.co/docs/transformers/en/main_classes/data_collator
    collate_fn=default_data_collator 
)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

model, train_dataloader, optimizer = accelerator.prepare(model, train_dataloader, optimizer)

# train config
num_epochs = 3
gradient_accumulation_steps = 4
total_steps = (len(train_dataloader) // gradient_accumulation_steps) * num_epochs
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)



# Initialize the progress bar and timer
progress_bar = tqdm(total=total_steps, desc="Training", position=0, dynamic_ncols=True)
start_time = time.time()
global_step = 0

model.train()





for epoch in range(num_epochs):
    print(f"🚀 Starting epoch {epoch+1}/{num_epochs}")

    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                labels=batch["labels"].to(device)
            )
            loss = outputs.loss
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                progress_bar.update(1)

                if accelerator.is_main_process:
                    # estimate the training time
                    elapsed = time.time() - start_time
                    eta = (elapsed / global_step) * (total_steps - global_step)

                    # wandb record
                    wandb.log({
                        "loss": loss.item(),
                        "step": global_step,
                        "lr": optimizer.param_groups[0]["lr"],
                        "ETA_min": eta / 60
                    })

                    # tqdm update
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "ETA": f"{eta / 60:.1f} min"
                    })

        # Forced exit condition(for unpredictable bug)
        if global_step >= total_steps:
            print("🛑 Reached total_steps, breaking inner loop.")
            break

    if global_step >= total_steps:
        print("🛑 Reached total_steps, breaking outer loop.")
        break
             

                   
print("over")
progress_bar.close()

# save the adapter
if accelerator.is_main_process:
    # print the current step
    print(f"🚨 Training completed: step {global_step}/{total_steps}")

    # make the output dic(if not existed)
    os.makedirs(output_dir, exist_ok=True)

    # save the adapter & tokenizer
    print("🧠 Saving the adapter and tokenizer...")
    accelerator.unwrap_model(model).save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Adapter and tokenizer saved to: {output_dir}")

    # Safely close wandb to prevent logging hang
    try:
        wandb.finish()
        print("📦 wandb logging finished.")
    except Exception as e:
        print(f"⚠️ wandb.finish() failed: {e}")