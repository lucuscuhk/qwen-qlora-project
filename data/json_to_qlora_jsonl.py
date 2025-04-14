import json

input_path = "data/100k_raw.json"
output_path = "data/100k_qllora_format.jsonl"

with open(input_path, "r") as f:
    raw_data = json.load(f)

with open(output_path, "w") as fout:
    for example in raw_data:
        prompt = example.get("user", "").strip()
        response = example.get("assistant", "").strip()
        new_example = {
            "instruction": prompt,
            "input": "",
            "output": response
        }
        fout.write(json.dumps(new_example, ensure_ascii=False) + "\n")

print(f"✅ Successfully converted {len(raw_data)} samples to QLoRA format.")