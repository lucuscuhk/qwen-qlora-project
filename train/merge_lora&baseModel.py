"""
merge_adapter_qwen2.5-7b.py

This script merges a LoRA adapter into its base Qwen2.5-7B-Instruct model to produce
a standalone, deployable model checkpoint. It loads the adapter and base model,
applies the LoRA weights using `merge_and_unload()`, and saves the full model and tokenizer
for downstream inference or deployment.

✅ Use Case:
After QLoRA fine-tuning, you typically save only the adapter weights to reduce storage.
This script lets you combine those adapter weights with the base model to produce a full model
compatible with `from_pretrained()` for inference.

🔧 Configuration:
- Adapter path: ./adapter&tokenizer_qwen2.5-7b
- Output path: ./Qlora_qwen2.5-7b-finance
- Merged model saved via `merged_model.save_pretrained(...)`
- Tokenizer loaded from base model and saved to the same directory

⚠️ Note:
Make sure the adapter and base model match (i.e., trained LoRA adapter was applied to the same base model).

Author: [Your Name]
Date: 2025-04-13
"""


from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

import os

current_dir = os.path.dirname(os.path.abspath(__file__))
adapter_path = os.path.join(current_dir, "..", "adapter&tokenizer_qwen2.5-7b")
adapter_path = os.path.abspath(adapter_path)      # saved directory of adpter
save_path = "../Qlora_qwen2.5-7b-finance"             # saved directory after merging adpter and base model

# load the adpter config and find the base model
peft_config = PeftConfig.from_pretrained(adapter_path)
base_model_path = peft_config.base_model_name_or_path

# load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    device_map="auto"
)

# load adapter
model = PeftModel.from_pretrained(base_model, adapter_path)

# merge adapter to base
merged_model = model.merge_and_unload()

# save the whole model for deploying the model
merged_model.save_pretrained(save_path, safe_serialization=True)

# save the tokenizer (the original tokenizer)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.save_pretrained(save_path)

print(f"✅ The merged model is saved successfully:{save_path}")