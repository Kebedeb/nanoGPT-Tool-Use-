"""
Training configuration for tool-use (basic format).

This config trains on addition problems with simple tool annotations.
"""

# I/O
out_dir = 'out-addition-scratchpad'
eval_interval = 250
eval_iters = 200
log_interval = 10

# wandb logging
wandb_log = False  # Set to True if you want to use Weights & Biases
wandb_project = 'nanogpt-tool-use'
wandb_run_name = 'addition-scratchpad'

# data
dataset = 'addition_tools'
train_bin = 'data/addition_tools/train_scratchpad.bin'
val_bin = 'data/addition_tools/val_scratchpad.bin'
meta_path = 'data/addition_tools/train_scratchpad_meta.pkl'

gradient_accumulation_steps = 1
batch_size = 64
block_size = 128  # Context length

# model - small for quick experiments
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1

# adamw optimizer
learning_rate = 1e-3
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# learning rate decay
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-4

# system
device = 'cpu'  # 'cpu', 'cuda', 'cuda:0', etc.
dtype = 'float32'
compile = False  # MUST be False for CPU