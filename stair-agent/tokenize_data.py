# tokenize_data.py
from transformers import AutoTokenizer

# 1. Load your base tokenizer (GPT-2 is standard for nanoGPT architectures)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2. Define your exact STAIR tags as special tokens
special_tokens_dict = {
    "additional_special_tokens": [
        "<tool:calc>", 
        "</tool:calc>", 
        "<lookup:calculations>",
        "[TABLE OF CONTENTS]",
        "[END TABLE OF CONTENTS]",
        "[CALCULATIONS CONTENT]:"
    ]
}

# 3. Add them to the tokenizer vocabulary
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Added {num_added_toks} special tokens to the vocabulary.")

# 4. Test it out to prove it works
test_sentence = "Model: <tool:calc>50 + 25</tool:calc>"
tokens = tokenizer.tokenize(test_sentence)
print(f"Tokenized output: {tokens}")