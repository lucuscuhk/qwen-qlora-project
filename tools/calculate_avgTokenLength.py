"""
calculate_avgTokenLength.py

This script mainly used for calculate the average token
length of the train_dataset, which is a helpful tool to
help determine the max_length in the training process


Author: [Lucus]
Date: 2025-04-13


"""
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("../Qwen2.5-7B-Instruct", trust_remote_code=True)

total_tokens = 0
num_samples = 0
max_samples = 10000  

with open("../data/100k_qllora_format.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        instruction = data.get("instruction", "")
        input_text = data.get("input", "")
        output = data.get("output", "")
        if input_text:
            prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        full_text = prompt + output

        tokens = tokenizer(full_text, return_tensors="pt").input_ids[0]
        total_tokens += len(tokens)
        num_samples += 1

        if num_samples >= max_samples:
            break

avg_tokens = total_tokens / num_samples
print(f"✅ Sampled {num_samples} examples")
print(f"📊 average tokens: {avg_tokens:.2f}")