out_dir = 'out-addition-tool'
eval_interval = 50
eval_iters = 20
log_interval = 10

always_save_checkpoint = True

dataset = 'addition_tool'
init_from = 'scratch'

batch_size = 64
block_size = 128

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

learning_rate = 1e-3
max_iters = 1000
lr_decay_iters = 1000
min_lr = 1e-4

beta2 = 0.99
warmup_iters = 100

device = 'cpu'  # change to 'cpu' if needed
compile = False