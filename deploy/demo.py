"""
demo.py

Gradio UI for interacting with a Qwen2.5-7B-Instruct model fine-tuned for financial domain Q&A.

Author: [Lucus]
Date: 2025-04-14
"""

import gradio as gr
from inference_utils import generate_response_stream  # Import streaming generation function

# Chat handler function
def chat_interface(user_input, history=None):
    if history is None:
        history = []

    stream = generate_response_stream(user_input, history)
    final_reply = ""

    try:
        for token in stream:
            final_reply = token
            yield final_reply  # Stream response to frontend
    finally:
        history.append((user_input, final_reply))  # Append interaction to history

# Build Gradio UI
gr.ChatInterface(
    fn=chat_interface,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(
        placeholder="Enter your finance-related question...", 
        container=False, 
        show_label=False
    ),
    title="📉 Qwen Finance ChatBot",
    theme="soft",
    examples=[
        "Analyze current A-share market trends",
        "Predict future RMB exchange rate movements",
        "Explain the relationship between GDP and CPI"
    ],
    cache_examples=False,
).launch(share=True)