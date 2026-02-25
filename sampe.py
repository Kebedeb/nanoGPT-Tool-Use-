"""
Tool-enabled sampling from a trained nanoGPT model
"""

import os
import pickle
import torch
import re
import calculator 
from contextlib import nullcontext
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# CONFIGURATION
init_from = 'resume'
out_dir = 'out-addition-tool'   # CHANGE if needed
device = 'cuda' if torch.cuda.is_available() else 'cpu'
temperature = 1.0               # IMPORTANT: keep deterministic
max_new_tokens = 200
seed = 1337
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device_type = 'cuda' if 'cuda' in device else 'cpu'
ctx = nullcontext()

# -----------------------------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------------------------

ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)

gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()
model.to(device)

# -----------------------------------------------------------------------------
# LOAD META (CHAR ENCODING)
# -----------------------------------------------------------------------------

dataset = checkpoint['config']['dataset']
meta_path = os.path.join('data', dataset, 'meta.pkl')

print(f"Loading meta from {meta_path}")

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# -----------------------------------------------------------------------------
# TOOL LOOP
# -----------------------------------------------------------------------------

def run_tool_loop():
    contGen = False
    while not contGen:
        prompt = input("\nEnter expression (e.g. 84321+99991=):\n")

        if not prompt.endswith("\n"):
            prompt += "\n"

        generated = prompt

        x = torch.tensor(encode(generated), dtype=torch.long, device=device)[None, ...]

        print("\nGenerating CALL...\n")

        # -------------------------------------------------
        # STEP 1: Generate until </CALL>
        # -------------------------------------------------

        for _ in range(100):

            x_cond = x[:, -model.config.block_size:]

            with torch.no_grad():
                logits, _ = model(x_cond)

                logits = logits[:, -1, :]
                logits = logits.float()

                # stability trick
                logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]

                probs = torch.softmax(logits, dim=-1)

                probs = torch.nan_to_num(probs, nan=0.0)

                if torch.any(probs < 0) or torch.any(torch.isnan(probs)):
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    idx_next = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, idx_next), dim=1)

            new_token = idx_next.item()
            new_char = decode([new_token])
            generated += new_char

            print(new_char, end="", flush=True)

            if "</CALL>" in generated:
                break

            if len(generated) > 500:
                print("\nGeneration too long, stopping.")
                return

        # -------------------------------------------------
        # STEP 2: Parse tool call
        # -------------------------------------------------

        print("\n\nParsing tool call...")

        match = re.search(r"add\(a=(\d+),b=(\d+)\)", generated)

        if match is None:
            print("Failed to parse tool call.")
            print("Generated text was:")
            print(generated)
            return

        a = int(match.group(1))
        b = int(match.group(2))

        print(f"Tool call detected: add({a}, {b})")

        # -------------------------------------------------
        # STEP 3: Execute tool
        # -------------------------------------------------

        result = calculator.add(a,b)
        # result = a+b
        print(f"Tool result: {result}")

        # -------------------------------------------------
        # STEP 4: Inject result
        # -------------------------------------------------

        injection = f"\n<RESULT>{result}</RESULT>\n"
        generated += injection

        x = torch.tensor(encode(generated), dtype=torch.long, device=device)[None, ...]

        # -------------------------------------------------
        # STEP 5: Continue generation
        # -------------------------------------------------

        print("\nContinuing generation...\n")

        for _ in range(50):

            x_cond = x[:, -model.config.block_size:]

            with torch.no_grad():
                logits, _ = model(x_cond)
                logits = logits[:, -1, :]
                logits = logits.float()

                logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]

                probs = torch.softmax(logits / temperature, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0)

                if torch.any(probs < 0) or torch.any(torch.isnan(probs)):
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    idx_next = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, idx_next), dim=1)

            generated = decode(x[0].tolist())

            # STOP after finishing result
            if "</RESULT>" in generated:
                break

                x = torch.cat((x, idx_next), dim=1)

        final_output = decode(x[0].tolist())

        print("\n================ FINAL OUTPUT ================\n")
        print(final_output)
        print("\n==============================================\n")
        response = input("Would you like to continue: ")
        if response.lower() == "no":
            contGen = True

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    run_tool_loop()