"""
data_split.py

This script randomly samples a subset of data from a jsonl file.
It is typically used to create a smaller dataset for training & 
testing.

Input:  a jsonl file with large amount of data
Output: a jsonl file with smaller amount of data

Example:
    Input file:  data/100k_qllora_format.jsonl
    Output file: data/30k_qllora_format.jsonl
    Subset size: 30,000 lines

Author: [Lucus]
Date: 2025-04-13
"""
import random


input_file = "data/100k_qllora_format.jsonl"
output_file = "data/30k_qllora_format.jsonl"
subset_size = 30000

with open(input_file, "r", encoding="utf-8") as f:
    all_lines = f.readlines()


subset_lines = random.sample(all_lines, subset_size)


with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(subset_lines)

print(f"✅ Done: Extracted {subset_size} samples from {input_file} to {output_file}")