# Fine-tuning configuration for adapting Shakespeare model to Linux kernel code
# This config will be overridden by the orchestration script for different experiments

import time

out_dir = 'out-finetune-kernel'
eval_interval = 100
eval_iters = 200
log_interval = 10

# Only save checkpoints if validation loss improves
always_save_checkpoint = False

# WandB logging
wandb_log = True
wandb_project = 'shakespeare-to-kernel-finetune'
wandb_run_name = 'finetune-' + str(time.time())

# Dataset - will be overridden by orchestration script
dataset = '/nobackup/gaurav/kernel_code/100k'

# Initialize from pre-trained Shakespeare checkpoint
init_from = 'resume'  # Will load from out_dir/ckpt.pt

# Batch settings - optimized for 8 GPUs
batch_size = 64
gradient_accumulation_steps = 8  # Must be divisible by num GPUs
block_size = 256  # Match pretrained model

# Model architecture will be loaded from checkpoint (L4-H4-E256)
# No need to specify n_layer, n_head, n_embd - they come from checkpoint

# Fine-tuning hyperparameters
learning_rate = 3e-4  # Small learning rate for fine-tuning
max_iters = 1000  # Will be overridden by orchestration script
lr_decay_iters = 1000  # Match max_iters
min_lr = 3e-5  # learning_rate / 10
beta2 = 0.99
warmup_iters = 50  # Short warmup for fine-tuning

# Use learning rate decay for fine-tuning
decay_lr = True

# Dropout for fine-tuning (helps prevent overfitting)
dropout = 0.1

# System
bias = False
compile = False  # Disable for compatibility

