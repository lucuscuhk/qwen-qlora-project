#!/bin/bash

# Hugging Face repo ID 
REPO_ID="lucus112/QLoRA_qwen2.5-7b-finance"

# Local target folder 
TARGET_DIR="./QLoRA_qwen2.5-7b-finance"

# Create target directory if it doesn't exist
mkdir -p $TARGET_DIR

# Download model from Hugging Face
huggingface-cli download $REPO_ID \
  --local-dir $TARGET_DIR \
  --local-dir-use-symlinks False

# Check if download succeeded
if [ $? -eq 0 ]; then
  echo "✅ Model downloaded successfully to: $TARGET_DIR"
else
  echo "❌ Download failed. Please check your network or the Hugging Face repo ID."
fi