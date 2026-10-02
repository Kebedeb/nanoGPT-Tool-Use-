import json
import numpy as np
from transformers import AutoTokenizer

# 1. Setup Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
special_tokens_dict = {
    "additional_special_tokens": [
        "<tool:calc>", "</tool:calc>", "<lookup:calculations>",
        "[TABLE OF CONTENTS]", "[END TABLE OF CONTENTS]", "[CALCULATIONS CONTENT]:"
    ]
}
tokenizer.add_special_tokens(special_tokens_dict)

# 2. Read your generated dataset
with open("stair_train.jsonl", "r") as f:
    lines = f.readlines()

# 3. Encode text to integers
all_token_ids = []
for line in lines:
    data = json.loads(line)
    # Encode the text and add an End-Of-Text token (50256 is standard for GPT-2)
    ids = tokenizer.encode(data["text"]) + [tokenizer.eos_token_id]
    all_token_ids.extend(ids)

# 4. Save to a binary file for training
token_array = np.array(all_token_ids, dtype=np.uint16)
token_array.tofile("train.bin")

print(f"Successfully encoded {len(token_array)} tokens and saved to 'train.bin'.")