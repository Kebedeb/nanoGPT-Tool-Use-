import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from model import GPTConfig, GPT

# --- CONFIG OVERRIDES FOR CPU ---
out_dir = 'out'
eval_interval = 50       # More frequent updates for CPU
log_interval = 1
eval_iters = 20
dataset = 'openwebtext'
gradient_accumulation_steps = 1 # No need to simulate larger batches on CPU
batch_size = 4           # Small batch size for CPU memory
block_size = 64          # Shorter sequences to speed up training
n_layer = 4              # Tiny model for CPU
n_head = 4
n_embd = 128
dropout = 0.0
learning_rate = 1e-3
max_iters = 500
device = 'cpu'           # FORCE CPU
compile = False         # MUST BE FALSE ON CPU
dtype = 'float32'       # Standard for CPU
# -------------------------------

# Logic to handle custom paths properly using globals()
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config_file = os.environ.get("NANOGPT_CONFIG", "configurator.py")
if os.path.exists(config_file):
    exec(open(config_file).read())
config = {k: globals()[k] for k in config_keys}

# Set seed
torch.manual_seed(1337)
device_type = 'cpu'
ptdtype = torch.float32
ctx = nullcontext() # No autocast needed for CPU

# Updated get_batch for CPU
data_dir = os.path.join('data', dataset)
train_bin_path = globals().get('train_bin', os.path.join(data_dir, 'train.bin'))
val_bin_path = globals().get('val_bin', os.path.join(data_dir, 'val.bin'))

def get_batch(split):
    bin_path = train_bin_path if split == 'train' else val_bin_path
    data = np.memmap(bin_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# Load vocab size
meta_file = globals().get('meta_path', os.path.join(data_dir, 'meta.pkl'))
meta_vocab_size = None
if os.path.exists(meta_file):
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']

# Model Init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=False, vocab_size=meta_vocab_size if meta_vocab_size else 50304, dropout=dropout)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

# Optimizer
optimizer = model.configure_optimizers(1e-1, learning_rate, (0.9, 0.95), device_type)

# Training Loop
X, Y = get_batch('train')
t0 = time.time()
iter_num = 0

while iter_num <= max_iters:
    # Basic LR schedule
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    # Forward/Backward
    with ctx:
        logits, loss = model(X, Y)
    
    X, Y = get_batch('train') # Prefetch next
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Timing and Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item()
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    
    iter_num += 1