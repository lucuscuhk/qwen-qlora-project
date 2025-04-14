"""
qwen2_module_inspect.py

This script loads the architecture of the Qwen2.5-7B-Instruct model in empty mode 
(i.e., without allocating GPU memory) and prints out all module names and their classes,
mainly used for detecting which layer to insert adapter


Author: [Lucus]
Date: 2025-04-13
"""

from transformers import AutoModelForCausalLM
from accelerate import init_empty_weights
import os

# Convert to an absolute path to prevent it from being mistaken for repo_id (avoid starting from ".")
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "..", "Qwen2.5-7B-Instruct")
model_dir = os.path.abspath(model_dir)  

# Load model architecture with zero memory allocation
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        device_map=None  # Do not assign to any device (we're just inspecting)
    )

# Print out all module names and their class types
for name, module in model.named_modules():
    print(name, ":", module.__class__.__name__)