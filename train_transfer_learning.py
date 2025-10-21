"""
Transfer learning script for fine-tuning across different vocabularies.

This script properly handles vocabulary mismatch when fine-tuning from
Shakespeare (small vocab) to kernel code (larger vocab) by:
1. Loading the pre-trained checkpoint
2. Creating a new model with the target vocabulary size
3. Copying compatible weights (transformer blocks)
4. Randomly initializing new embeddings
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# Import the standard training script's configuration
exec(open('configurator.py').read())

# Additional config for transfer learning
source_checkpoint = 'out-shakespeare-pretrained/ckpt.pt'  # Source checkpoint path
freeze_transformer = False  # Whether to freeze transformer weights during fine-tuning

# -----------------------------------------------------------------------------

def load_checkpoint_for_transfer(source_checkpoint, target_vocab_size, device='cuda'):
    """
    Load a checkpoint and adapt it for a different vocabulary size.
    
    Args:
        source_checkpoint: Path to source checkpoint
        target_vocab_size: Vocabulary size of target dataset
        device: Device to load checkpoint on
    
    Returns:
        Tuple of (model, iter_num, best_val_loss)
    """
    print(f"Loading checkpoint from {source_checkpoint} for transfer learning...")
    
    checkpoint = torch.load(source_checkpoint, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    
    source_vocab_size = checkpoint_model_args['vocab_size']
    print(f"  Source vocab size: {source_vocab_size}")
    print(f"  Target vocab size: {target_vocab_size}")
    
    if source_vocab_size == target_vocab_size:
        print("  ✓ Vocabularies match, using standard resume")
        # Standard resume
        gptconf = GPTConfig(**checkpoint_model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        
        # Fix key names
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict)
        
    else:
        print("  ⚠ Vocabulary mismatch detected, performing transfer learning")
        
        # Create new model with target vocab size
        new_model_args = checkpoint_model_args.copy()
        new_model_args['vocab_size'] = target_vocab_size
        
        print(f"  Creating new model with vocab_size={target_vocab_size}")
        gptconf = GPTConfig(**new_model_args)
        model = GPT(gptconf)
        
        # Load source state dict
        source_state_dict = checkpoint['model']
        
        # Fix key names
        unwanted_prefix = '_orig_mod.'
        for k, v in list(source_state_dict.items()):
            if k.startswith(unwanted_prefix):
                source_state_dict[k[len(unwanted_prefix):]] = source_state_dict.pop(k)
        
        # Copy compatible weights
        model_state_dict = model.state_dict()
        transferred_keys = []
        skipped_keys = []
        
        for key, value in source_state_dict.items():
            # Skip embedding layers (vocabulary-dependent)
            if 'wte' in key or 'lm_head' in key:
                skipped_keys.append(key)
                print(f"    Skip (vocab-dependent): {key}")
                continue
            
            # Copy compatible weights
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    model_state_dict[key] = value
                    transferred_keys.append(key)
                else:
                    skipped_keys.append(key)
                    print(f"    Skip (shape mismatch): {key}")
            else:
                skipped_keys.append(key)
                print(f"    Skip (not in target): {key}")
        
        # Load the modified state dict
        model.load_state_dict(model_state_dict)
        
        print(f"\n  ✓ Transferred {len(transferred_keys)} layers")
        print(f"  ✓ Skipped {len(skipped_keys)} layers (will be randomly initialized)")
        print(f"  ✓ New embeddings initialized for vocab_size={target_vocab_size}")
    
    # Don't resume iteration number or best_val_loss for transfer learning
    # We want to start fresh training
    iter_num = 0
    best_val_loss = float('inf')
    
    return model, iter_num, best_val_loss


# -----------------------------------------------------------------------------
# Main training code (simplified version of train.py)

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data loading
data_dir = os.path.join('data', dataset) if not os.path.isabs(dataset) else dataset
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Get target vocabulary size from meta.pkl
meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    target_vocab_size = meta['vocab_size']
    if master_process:
        print(f"Target dataset vocab size: {target_vocab_size}")
else:
    raise FileNotFoundError(f"meta.pkl not found at {meta_path}")

# Transfer learning: Load checkpoint and adapt to new vocabulary
model, iter_num, best_val_loss = load_checkpoint_for_transfer(
    source_checkpoint=source_checkpoint,
    target_vocab_size=target_vocab_size,
    device=device
)

# Optionally freeze transformer blocks
if freeze_transformer:
    if master_process:
        print("Freezing transformer blocks (only training embeddings)")
    for name, param in model.named_parameters():
        if 'wte' not in name and 'lm_head' not in name:
            param.requires_grad = False

# Move model to device
model.to(device)

# Initialize a GradScaler
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# Compile model
if compile:
    if master_process:
        print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# Wrap in DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Training/validation function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Learning rate decay scheduler
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Training loop
if master_process:
    print(f"\n{'='*70}")
    print(f"STARTING TRANSFER LEARNING")
    print(f"{'='*70}")
    print(f"Source: {source_checkpoint}")
    print(f"Target dataset: {dataset}")
    print(f"Max iterations: {max_iters}")
    print(f"{'='*70}\n")

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    # Learning rate schedule
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
            })
        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': {
                        'n_layer': raw_model.config.n_layer,
                        'n_head': raw_model.config.n_head,
                        'n_embd': raw_model.config.n_embd,
                        'block_size': raw_model.config.block_size,
                        'bias': raw_model.config.bias,
                        'vocab_size': raw_model.config.vocab_size,
                        'dropout': raw_model.config.dropout,
                    },
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    
    if iter_num == 0 and eval_only:
        break

    # Forward backward update
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

