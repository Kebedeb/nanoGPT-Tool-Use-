import os
import pickle
import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")

# 1. Define custom tokens and expand vocab
special_tokens = ["<|tool_call|>", "</|tool_call|>", "<|tool_response|>", "</|tool_response|>"]
custom_vocab = {token: 50257 + i for i, token in enumerate(special_tokens)}
vocab_size = 50257 + len(special_tokens)

# 2. Format a trajectory
prompt_ids = enc.encode_ordinary("What is 15 plus 27?")
thought_ids = enc.encode_ordinary("I need to add these numbers together. ")
call_ids = [custom_vocab["<|tool_call|>"]] + enc.encode_ordinary("15+27") + [custom_vocab["</|tool_call|>"]]
response_ids = [custom_vocab["<|tool_response|>"]] + enc.encode_ordinary("42") + [custom_vocab["</|tool_response|>"]]
answer_ids = enc.encode_ordinary(" The answer is 42.")

# 3. Concatenate and shift for targets
x = prompt_ids + thought_ids + call_ids + response_ids + answer_ids
y = x[1:] + [enc.encode_ordinary("<|endoftext|>")[0]] 

# 4. Apply the -100 mask to targets the model shouldn't be penalized for
mask = (
    [-100] * len(prompt_ids) + 
    thought_ids + 
    call_ids + 
    [-100] * len(response_ids) + 
    answer_ids
)

for i in range(len(y)):
    if mask[i] == -100:
        y[i] = -100

# 5. Duplicate into a stream and save as int32
x_stream = x * 1000
y_stream = y * 1000

os.makedirs('data/tools', exist_ok=True)
np.array(x_stream, dtype=np.int32).tofile('data/tools/train_x.bin')
np.array(y_stream, dtype=np.int32).tofile('data/tools/train_y.bin')

with open('data/tools/meta.pkl', 'wb') as f:
    pickle.dump({'vocab_size': vocab_size}, f)