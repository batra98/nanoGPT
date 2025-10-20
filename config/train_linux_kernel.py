# Train nanoGPT on Linux kernel source code
# Uses best architecture from Shakespeare experiment: L4-H4-E256
# With early stopping for efficiency

out_dir = 'out-linux-kernel'
eval_interval = 250
eval_iters = 200
log_interval = 10

# Save checkpoint when validation improves
always_save_checkpoint = False

# WandB logging
wandb_log = True
wandb_project = 'linux-kernel-data-scaling'
wandb_run_name = 'linux-kernel'  # Will be overridden by orchestration script

# Dataset - will be overridden by orchestration script to point to specific subset
dataset = 'linux_kernel/100k'

# Training parameters
gradient_accumulation_steps = 8  # For DDP with 8 GPUs
batch_size = 64
block_size = 256  # Context of up to 256 previous characters

# Best architecture from Part 1: L4-H4-E256
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.2

# Learning rate schedule
learning_rate = 1e-3
max_iters = 10000  # Will stop early if no improvement
lr_decay_iters = 10000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

# Early stopping parameters
early_stopping_patience = 5
early_stopping_min_delta = 0.001

