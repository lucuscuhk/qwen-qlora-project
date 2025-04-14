"""
inference_utils.py

This script loads the Qwen2.5-7B-Instruct model with 4-bit quantization (QLoRA setup),
and provides a utility to stream responses from the model based on user input and conversation history.

Note: This script is intended for backend logic or API integration. 
For full UI use, launch demo.py

Author: [Lucus]
Date: 2025-04-14
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
import threading, os

# Convert to absolute path to avoid being treated as a repo_id
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "..", "Qwen2.5-7B-Instruct")
model_path = os.path.abspath(model_dir)

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config
)
model.eval()

# Build prompt from user input and chat history
def build_prompt(user_input, history):
    prompt = ""
    for past_user, past_assistant in history:
        prompt += f"<|im_start|>user\n{past_user}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{past_assistant}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

# Streamed text generation
def generate_response_stream(user_input, history):
    prompt = build_prompt(user_input, history)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
        eos_token_id=151643
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_reply = ""
    for new_text in streamer:
        partial_reply += new_text
        yield partial_reply