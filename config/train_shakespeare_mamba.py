# Training configuration for Mamba model on Shakespeare
# Uses the same hyperparameters as the best Transformer config (L4-H4-E256)
# from the extensive hyperparameter search for fair comparison

out_dir = 'out-shakespeare-mamba'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

# WandB logging
wandb_log = True
wandb_project = 'shakespeare-mamba-comparison'
wandb_run_name = 'mamba-L4-D256-S16'

# Dataset
dataset = 'shakespeare_char'
gradient_accumulation_steps = 8  # Set to 8 for 8 GPUs (or adjust based on available GPUs)
batch_size = 64
block_size = 256  # context of up to 256 previous characters

# Model architecture - matching best Transformer config
n_layer = 4      # Same as best Transformer
n_head = 4       # Kept for compatibility (not used in Mamba)
n_embd = 256     # Same embedding dimension as best Transformer
dropout = 0.0    # No dropout

# Mamba-specific parameters
d_state = 16  # SSM state dimension
d_conv = 4    # Convolution kernel size

# Training hyperparameters - matching baseline
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# System
bias = False
compile = False  # Disable torch.compile for compatibility with custom Mamba layer

