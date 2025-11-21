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
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print("Training Reward Model")
    print(f"{'='*60}")
    print(f"GPT checkpoint: {args.gpt_checkpoint}")
    print(f"Train data: {args.train_data}")
    print(f"Val data: {args.val_data}")
    print(f"Output directory: {args.out_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Block size: {args.block_size}")
    print(f"Num epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")
    
    # Load reward model
    print("Loading reward model...")
    reward_model = create_reward_model(
        args.gpt_checkpoint,
        device=args.device,
        freeze_gpt=args.freeze_gpt
    )
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = PreferenceDataset(args.train_data)
    val_dataset = PreferenceDataset(args.val_data)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, args.block_size)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, args.block_size)
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Optimizer (only for reward head parameters)
    optimizer = torch.optim.AdamW(
        [p for p in reward_model.parameters() if p.requires_grad],
        lr=args.learning_rate
    )
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")
    
    best_val_accuracy = 0
    
    for epoch in range(args.num_epochs):
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
        
        print(f"Epoch {epoch+1}/{args.num_epochs} ({dt:.2f}s):")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': reward_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.out_dir, 'best_model.pt'))
            print(f"  → Saved best model (val_acc={val_acc:.4f})")
        
        print()
    
    print(f"{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Model saved to: {args.out_dir}/best_model.pt")


if __name__ == '__main__':
    main()

