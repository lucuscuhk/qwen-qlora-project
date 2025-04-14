from datasets import load_dataset

dataset = load_dataset("json", data_files="../data/100k_qllora_format.jsonl")
dataset = dataset["train"]
print(dataset[0])