# Base configuration for hyperparameter search on Shakespeare character-level model
# Designed to complete in <10 minutes on RTX 2080 Ti

out_dir = 'out-shakespeare-hyperparam'  # Will be overridden by run script
eval_interval = 500  # Evaluate less frequently to save time
eval_iters = 100     # Fewer eval iterations to speed up
log_interval = 50    # Log periodically

# Save only the best checkpoint to save disk space
always_save_checkpoint = False

# WandB logging - will be configured by orchestration script
wandb_log = True
wandb_project = 'shakespeare-hyperparam-search'
wandb_run_name = 'baseline'  # Will be overridden

# Dataset
dataset = 'shakespeare_char'
gradient_accumulation_steps = 8  # Set to 8 for 8-GPU training (must be divisible by num GPUs)
batch_size = 64
block_size = 256  # context of up to 256 previous characters

# Model hyperparameters - will be overridden by orchestration script
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# Training hyperparameters
learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 3500      # Reduced from 5000 to ensure <10 min runtime
lr_decay_iters = 3500 # make equal to max_iters
min_lr = 1e-4         # learning_rate / 10
beta2 = 0.99          # make a bit bigger because number of tokens per iter is small

warmup_iters = 100

# Compilation and device settings (can be overridden via command line)
# compile = True  # PyTorch 2.0 compile for speedup
# device = 'cuda'

