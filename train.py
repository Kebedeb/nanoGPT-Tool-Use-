import os
import time
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from model import GPTConfig, GPT

# --- 1. HARDCODED CPU CONFIG (Bypasses configurator.py) ---
out_dir = 'out'
eval_interval = 50
log_interval = 1
eval_iters = 20
dataset = 'addition_tools' 
batch_size = 4
block_size = 64
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
learning_rate = 1e-3
max_iters = 500
device = 'cpu'
dtype = 'float32'
# ---------------------------------------------------------

# Setup directories and seeds
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
device_type = 'cpu'
ctx = nullcontext()

# Data handling
data_dir = os.path.join('data', dataset)
# Check for custom bin paths, default to standard data_dir paths
train_bin_path = os.path.join(data_dir, 'train_basic.bin')
val_bin_path = os.path.join(data_dir, 'val.bin')

def get_batch(split):
    bin_path = train_bin_path if split == 'train' else val_bin_path
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Cannot find {bin_path}. Did you run prepare.py?")
    data = np.memmap(bin_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# Load vocab size from meta.pkl
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"Found metadata. Vocab size: {meta_vocab_size}")

# Initialize Model
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=False, vocab_size=meta_vocab_size if meta_vocab_size else 50304, dropout=dropout)

print("Initializing model on CPU...")
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

# Optimizer
optimizer = model.configure_optimizers(weight_decay=1e-1, learning_rate=learning_rate, betas=(0.9, 0.95), device_type='cpu')

# Training Loop
print("Starting training loop...")
X, Y = get_batch('train')
t0 = time.time()
iter_num = 0

while iter_num <= max_iters:
    # Forward/Backward
    with ctx:
        logits, loss = model(X, Y)
    
    # Simple step
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    # Fetch next batch
    X, Y = get_batch('train')

    # Timing and Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms")
    
    iter_num += 1