"""
Train reward model on preference pairs.

Uses pairwise ranking loss (Bradley-Terry model) to train the reward head
while keeping the GPT backbone frozen.
"""

import os
import time
import pickle
import argparse
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import numpy as np

from reward_model import create_reward_model


class PreferenceDataset(Dataset):
    """Dataset for preference pairs."""
    
    def __init__(self, data_path: str, encode_fn=None):
        """
        Load preference pairs from pickle file.
        
        Args:
            data_path: Path to .pkl file with preference data
            encode_fn: Optional encoding function (for text->tokens)
        """
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.encode_fn = encode_fn
        print(f"Loaded {len(self.data)} preference pairs from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # If completions are already tokenized (lists), use them
        # Otherwise encode them
        if isinstance(item['full_a'], str):
            # Need to encode
            if self.encode_fn is None:
                raise ValueError("Text data requires encode_fn")
            tokens_a = self.encode_fn(item['full_a'])
            tokens_b = self.encode_fn(item['full_b'])
        else:
            tokens_a = item['full_a']
            tokens_b = item['full_b']
        
        return {
            'tokens_a': torch.tensor(tokens_a, dtype=torch.long),
            'tokens_b': torch.tensor(tokens_b, dtype=torch.long),
            'preference': item['preference'],  # 0 if A preferred, 1 if B preferred
            'density_a': item['density_a'],
            'density_b': item['density_b']
        }


def collate_fn(batch, block_size=256):
    """
    Collate function to handle variable-length sequences.
    Pads/truncates to block_size.
    """
    tokens_a = []
    tokens_b = []
    preferences = []
    densities_a = []
    densities_b = []
    
    for item in batch:
        # Truncate or pad to block_size
        tok_a = item['tokens_a'][:block_size]
        tok_b = item['tokens_b'][:block_size]
        
        # Pad if necessary
        if len(tok_a) < block_size:
            tok_a = F.pad(tok_a, (0, block_size - len(tok_a)), value=0)
        if len(tok_b) < block_size:
            tok_b = F.pad(tok_b, (0, block_size - len(tok_b)), value=0)
        
        tokens_a.append(tok_a)
        tokens_b.append(tok_b)
        preferences.append(item['preference'])
        densities_a.append(item['density_a'])
        densities_b.append(item['density_b'])
    
    return {
        'tokens_a': torch.stack(tokens_a),
        'tokens_b': torch.stack(tokens_b),
        'preference': torch.tensor(preferences, dtype=torch.long),
        'density_a': torch.tensor(densities_a, dtype=torch.float),
        'density_b': torch.tensor(densities_b, dtype=torch.float)
    }


def compute_pairwise_loss(reward_a: torch.Tensor, reward_b: torch.Tensor, preference: torch.Tensor) -> torch.Tensor:
    """
    Compute Bradley-Terry pairwise ranking loss.
    
    Loss = -log(sigmoid(reward_preferred - reward_other))
    
    Args:
        reward_a: Rewards for completion A (batch_size,)
        reward_b: Rewards for completion B (batch_size,)
        preference: 0 if A preferred, 1 if B preferred (batch_size,)
    
    Returns:
        loss: Scalar loss
    """
    # For samples where A is preferred (preference=0), we want reward_a > reward_b
    # For samples where B is preferred (preference=1), we want reward_b > reward_a
    
    # Compute logits: positive when preferred is better
    logits = torch.where(
        preference == 0,
        reward_a - reward_b,  # A preferred: want positive difference
        reward_b - reward_a   # B preferred: want positive difference
    )
    
    # Bradley-Terry loss: -log(sigmoid(logits))
    loss = -F.logsigmoid(logits).mean()
    
    return loss


def train_epoch(model, dataloader, optimizer, device, block_size=256):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(dataloader):
        tokens_a = batch['tokens_a'].to(device)
        tokens_b = batch['tokens_b'].to(device)
        preference = batch['preference'].to(device)
        
        # Forward pass
        reward_a = model(tokens_a)
        reward_b = model(tokens_b)
        
        # Compute loss
        loss = compute_pairwise_loss(reward_a, reward_b, preference)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        predicted_preference = (reward_b > reward_a).long()
        correct += (predicted_preference == preference).sum().item()
        total += preference.size(0)
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


@torch.no_grad()
def eval_epoch(model, dataloader, device, block_size=256):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        tokens_a = batch['tokens_a'].to(device)
        tokens_b = batch['tokens_b'].to(device)
        preference = batch['preference'].to(device)
        
        # Forward pass
        reward_a = model(tokens_a)
        reward_b = model(tokens_b)
        
        # Compute loss
        loss = compute_pairwise_loss(reward_a, reward_b, preference)
        
        # Compute accuracy
        predicted_preference = (reward_b > reward_a).long()
        correct += (predicted_preference == preference).sum().item()
        total += preference.size(0)
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train reward model')
    parser.add_argument('--gpt_checkpoint', type=str, default='out-shakespeare/ckpt.pt',
                       help='Path to GPT checkpoint')
    parser.add_argument('--train_data', type=str, default='data/preferences/train.pkl',
                       help='Path to training preference data')
    parser.add_argument('--val_data', type=str, default='data/preferences/val.pkl',
                       help='Path to validation preference data')
    parser.add_argument('--out_dir', type=str, default='out-reward-model',
                       help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--block_size', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--freeze_gpt', action='store_true', default=True,
                       help='Freeze GPT weights (only train reward head)')
    parser.add_argument('--wandb_log', action='store_true', default=False,
                       help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='rlhf-reward-model',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name (default: auto-generated)')
    
    args = parser.parse_args()
    
    # DDP setup
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        args.device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(args.device)
        master_process = ddp_rank == 0
        # batch_size is per-GPU, total batch = batch_size * world_size
    else:
        master_process = True
        ddp_world_size = 1
    
    # Create output directory (only on master)
    if master_process:
        os.makedirs(args.out_dir, exist_ok=True)
    
    if master_process:
        print(f"{'='*60}")
        print("Training Reward Model")
        print(f"{'='*60}")
        print(f"GPT checkpoint: {args.gpt_checkpoint}")
        print(f"Train data: {args.train_data}")
        print(f"Val data: {args.val_data}")
        print(f"Output directory: {args.out_dir}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Total batch size: {args.batch_size * ddp_world_size if ddp else args.batch_size}")
        print(f"Block size: {args.block_size}")
        print(f"Num epochs: {args.num_epochs}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Device: {args.device}")
        print(f"DDP: {ddp}, World size: {ddp_world_size}")
        print(f"{'='*60}\n")
    
    # Load reward model
    if master_process:
        print("Loading reward model...")
    reward_model = create_reward_model(
        args.gpt_checkpoint,
        device=args.device,
        freeze_gpt=args.freeze_gpt
    )
    
    # Wrap in DDP if multi-GPU
    if ddp:
        reward_model = DDP(reward_model, device_ids=[ddp_local_rank])
        raw_model = reward_model.module
    else:
        raw_model = reward_model
    
    # Get encode function from checkpoint
    if master_process:
        print("\nLoading encode function...")
    checkpoint = torch.load(args.gpt_checkpoint, map_location='cpu')
    encode_fn = None
    
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:
        dataset = checkpoint['config']['dataset']
        meta_path = os.path.join('data', dataset, 'meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi = meta['stoi']
            encode_fn = lambda s: [stoi[c] for c in s]
            if master_process:
                print(f"Loaded character-level encoder from {meta_path}")
        else:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            encode_fn = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            if master_process:
                print("Using tiktoken GPT-2 BPE encoder")
    else:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode_fn = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        if master_process:
            print("Using tiktoken GPT-2 BPE encoder (default)")
    
    # Load datasets
    if master_process:
        print("\nLoading datasets...")
    train_dataset = PreferenceDataset(args.train_data, encode_fn=encode_fn)
    val_dataset = PreferenceDataset(args.val_data, encode_fn=encode_fn)
    
    # Create dataloaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if ddp else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False) if ddp else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=lambda b: collate_fn(b, args.block_size)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=lambda b: collate_fn(b, args.block_size)
    )
    
    if master_process:
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
    
    # Optimizer (only for reward head parameters)
    optimizer = torch.optim.AdamW(
        [p for p in reward_model.parameters() if p.requires_grad],
        lr=args.learning_rate
    )
    
    # Initialize wandb
    if args.wandb_log:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"reward-model-{int(time.time())}",
            config={
                'gpt_checkpoint': args.gpt_checkpoint,
                'batch_size': args.batch_size,
                'block_size': args.block_size,
                'num_epochs': args.num_epochs,
                'learning_rate': args.learning_rate,
                'freeze_gpt': args.freeze_gpt,
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
            }
        )
    
    # Training loop
    if master_process:
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}\n")
    
    best_val_accuracy = 0
    
    for epoch in range(args.num_epochs):
        if ddp:
            train_sampler.set_epoch(epoch)
        
        t0 = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            reward_model, train_loader, optimizer, args.device, args.block_size
        )
        
        # Validate
        val_loss, val_acc = eval_epoch(
            reward_model, val_loader, args.device, args.block_size
        )
        
        t1 = time.time()
        dt = t1 - t0
        
        if master_process:
            print(f"Epoch {epoch+1}/{args.num_epochs} ({dt:.2f}s):")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Log to wandb
            if args.wandb_log:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'train/accuracy': train_acc,
                    'val/loss': val_loss,
                    'val/accuracy': val_acc,
                    'time_per_epoch': dt,
                })
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'args': vars(args)
                }
                torch.save(checkpoint, os.path.join(args.out_dir, 'best_model.pt'))
                print(f"  → Saved best model (val_acc={val_acc:.4f})")
                
                if args.wandb_log:
                    wandb.log({'best_val_accuracy': best_val_accuracy})
            
            print()
    
    if master_process:
        print(f"{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Best validation accuracy: {best_val_accuracy:.4f}")
        print(f"Model saved to: {args.out_dir}/best_model.pt")
        
        if args.wandb_log:
            wandb.log({'final/best_val_accuracy': best_val_accuracy})
            wandb.finish()
    
    if ddp:
        destroy_process_group()


if __name__ == '__main__':
    main()

