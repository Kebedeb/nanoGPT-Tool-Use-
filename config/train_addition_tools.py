"""
Training configuration for tool-use experiments.

This config is optimized for CPU training on the addition-with-tools task.
"""

# Output directory for checkpoints
out_dir = 'out-addition-tools'

# How often to evaluate and log
eval_interval = 250  # Evaluate every 250 iterations
eval_iters = 200     # Use 200 batches for evaluation
log_interval = 10    # Log every 10 iterations

# Weights & Biases logging (optional)
wandb_log = False
wandb_project = 'nanogpt-tool-use'
wandb_run_name = 'addition-tools'

# Data paths
dataset = 'addition_tools'
gradient_accumulation_steps = 1
batch_size = 64      # Process 64 examples at a time
block_size = 128     # Context window of 128 characters

# Model architecture (small for CPU training)
n_layer = 4          # 4 transformer layers
n_head = 4           # 4 attention heads
n_embd = 128         # Embedding dimension of 128
dropout = 0.1        # 10% dropout for regularization

# Optimizer settings
learning_rate = 1e-3  # 0.001
max_iters = 5000      # Train for 5000 iterations
weight_decay = 1e-1   # 0.1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0       # Clip gradients to prevent explosion

# Learning rate schedule
decay_lr = True
warmup_iters = 100    # Warm up for 100 iterations
lr_decay_iters = 5000 # Decay over 5000 iterations
min_lr = 1e-4         # Minimum learning rate

# System settings
device = 'cpu'        # Use CPU (no GPU)
dtype = 'float32'     # Use 32-bit floats
compile = False       # MUST be False for CPU