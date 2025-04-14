"""
load_qwen2.5-7b-test.py

Test script for loading Qwen2.5-7B-Instruct with QLoRA (4-bit) and injecting LoRA adapters.

No training or inference is performed

Author: [Lucus]
Date: 2025-04-13
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch, os

# Convert to an absolute path to prevent it from being mistaken for repo_id (avoid starting from ".")
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "..", "Qwen2.5-7B-Instruct")
model_dir = os.path.abspath(model_dir)  

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    trust_remote_code=True,
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# LoRA config
lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Inject LoRA
model = get_peft_model(model, lora_config)

# Show trainable parameters
model.print_trainable_parameters()