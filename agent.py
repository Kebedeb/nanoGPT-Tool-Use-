import os
import torch
import tiktoken
from transformer_model import TinyTransformerLM
import calculator
import re 

# 1. Setup device and tokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = tiktoken.get_encoding("gpt2")

special_tokens = ["<|tool_call|>", "</|tool_call|>", "<|tool_response|>", "</|tool_response|>"]
custom_vocab = {token: 50257 + i for i, token in enumerate(special_tokens)}
inverse_custom_vocab = {v: k for k, v in custom_vocab.items()}

# Safe decoder helper for custom tokens
fn_decode = lambda ids: "".join([inverse_custom_vocab[i] if i in inverse_custom_vocab else enc.decode([i]) for i in ids])

# 2. Load the trained checkpoint
# ckpt_path = os.path.join('out', 'ckpt.pt')
# checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
# model_args = checkpoint.get('model_args', {})

# # Force valid integers even if the keys hold None in the checkpoint
# vocab_size = model_args.get('vocab_size') or 50261
# n_layers = model_args.get('n_layer') or 12
# n_heads = model_args.get('n_head') or 12
# block_size = model_args.get('block_size') or 1024
# n_embd = model_args.get('n_embd') or 768
# mlp_multiplier = model_args.get('mlp_multiplier') or 4

# model = TinyTransformerLM(
#     vocab_size=vocab_size,
#     n_layers=n_layers,
#     n_heads=n_heads,
#     block_size=block_size,
#     n_embd=n_embd,
#     mlp_multiplier=mlp_multiplier
# )
# 2. Load the trained checkpoint and force everything to device
ckpt_path = os.path.join('out', 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
model_args = checkpoint.get('model_args', {})

vocab_size = model_args.get('vocab_size') or 50261
n_layers = model_args.get('n_layer') or 12
n_heads = model_args.get('n_head') or 12
block_size = model_args.get('block_size') or 1024
n_embd = model_args.get('n_embd') or 768
mlp_multiplier = model_args.get('mlp_multiplier') or 4

# Initialize model
model = TinyTransformerLM(
    vocab_size=vocab_size,
    n_layers=n_layers,
    n_heads=n_heads,
    block_size=block_size,
    n_embd=n_embd,
    mlp_multiplier=mlp_multiplier
)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)

# ---> YEHA HAI ASLI JAADU <---
model.to(device)
for p in model.parameters():
    p.data = p.data.to(device)
for b in model.buffers():
    b.data = b.data.to(device)

model.eval()
# 3. Agent Execution Loop
def execute_agent_loop(prompt_text):
    print(f"DEBUG: Starting agent with prompt -> {prompt_text}")
    
    # Encode prompt using allowed_special to parse custom tags safely
    input_ids = enc.encode(prompt_text, allowed_special=set(special_tokens))
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        # Generate initial tokens up to the tool call
        idx = model.generate(idx, max_new_tokens=20)
        decoded_text = fn_decode(idx[0].tolist())
        print(f"\n[Phase 1 Generated]:\n{decoded_text}")

        # Check for tool call
        if "</|tool_call|>" in decoded_text and "<|tool_response|>" not in decoded_text:
            print("\n-> Intercepted tool call! Parsing equation...")
            raw_equation = decoded_text.split("<|tool_call|>")[-1].split("</|tool_call|>")[0].strip()
            
            # Extract only valid math characters (numbers and basic operators like +, -, *, /)
            equation = "".join(re.findall(r'[0-9.+\-*/]', raw_equation))
            print(f"-> Cleaned Equation: {equation}")

            # Execute calculator tool safely
            try:
                result = str(calculator.evaluate(equation))
            except Exception as e:
                result = f"Error: {e}"
            print(f"-> Calculator Result: {result}")

            # Format and inject response IDs
            response_ids = [custom_vocab["<|tool_response|>"]] + enc.encode(result) + [custom_vocab["</|tool_response|>"]]
            idx = torch.cat([idx, torch.tensor([response_ids], dtype=torch.long, device=device)], dim=1)

            # Resume generation for final answer (removed unsupported temperature/top_k)
            idx = model.generate(idx, max_new_tokens=20)
            final_output = fn_decode(idx[0].tolist())
            print(f"\n[Phase 2 Final Output]:\n{final_output}")
            return final_output
        else:
            print("\n-> No tool call triggered.")
            return decoded_text

if __name__ == "__main__":
    test_prompt = "What is 45 plus 55? <|tool_call|>45+55</|tool_call|>"
    execute_agent_loop(test_prompt)