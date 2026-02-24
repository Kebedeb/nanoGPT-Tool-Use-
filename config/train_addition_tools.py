"""
Training config for tool syntax learning.
"""

# Output
out_dir = 'out-addition-tools'
eval_interval = 250
eval_iters = 200
log_interval = 10

# Logging
wandb_log = False
wandb_project = 'tool-use'
wandb_run_name = 'addition-tools'

# Data
dataset = 'addition_tools'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 128

# Model (small for CPU)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1

# Optimizer
learning_rate = 1e-3
max_iters = 250
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# LR schedule
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-4

# System
device = 'cpu'
dtype = 'float32'
compile = False