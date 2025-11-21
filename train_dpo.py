"""
Train model using Direct Preference Optimization (DPO).

DPO directly optimizes the policy to align with preferences without
needing a separate reward model or RL loop.

Paper: https://arxiv.org/abs/2305.18290
"""

import os
import time
import pickle
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from model import GPTConfig, GPT


class PreferenceDataset(Dataset):
    """Dataset for preference pairs (for DPO)."""
    
    def __init__(self, data_path: str):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"Loaded {len(self.data)} preference pairs from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'tokens_w': torch.tensor(item['tokens_w'], dtype=torch.long),  # preferred (winner)
            'tokens_l': torch.tensor(item['tokens_l'], dtype=torch.long),  # less preferred (loser)
            'prompt_len': item['prompt_len']
        }


def collate_fn(batch, block_size=256):
    """Collate with padding/truncation."""
    tokens_w_list = []
    tokens_l_list = []
    prompt_lens = []
    
    for item in batch:
        tok_w = item['tokens_w'][:block_size]
        tok_l = item['tokens_l'][:block_size]
        
        if len(tok_w) < block_size:
            tok_w = F.pad(tok_w, (0, block_size - len(tok_w)), value=0)
        if len(tok_l) < block_size:
            tok_l = F.pad(tok_l, (0, block_size - len(tok_l)), value=0)
        
        tokens_w_list.append(tok_w)
        tokens_l_list.append(tok_l)
        prompt_lens.append(item['prompt_len'])
    
    return {
        'tokens_w': torch.stack(tokens_w_list),
        'tokens_l': torch.stack(tokens_l_list),
        'prompt_lens': torch.tensor(prompt_lens, dtype=torch.long)
    }


def get_log_probs(model, tokens, prompt_len=None):
    """
    Get log probabilities of tokens under the model.
    
    Args:
        model: GPT model
        tokens: Token IDs (batch_size, seq_len)
        prompt_len: Length of prompt (to only compute loss on completion)
    
    Returns:
        log_probs: Log probabilities (batch_size,)
    """
    # Forward pass
    logits, _ = model(tokens[:, :-1], targets=None)  # Predict next token
    
    # Get log probs
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Gather log probs of actual next tokens
    next_tokens = tokens[:, 1:]  # Shifted targets
    log_probs = torch.gather(log_probs, -1, next_tokens.unsqueeze(-1)).squeeze(-1)
    
    # Mask out prompt tokens if prompt_len is provided
    if prompt_len is not None:
        mask = torch.arange(log_probs.size(1), device=log_probs.device)[None, :] >= prompt_len[:, None]
        log_probs = log_probs * mask.float()
        # Sum log probs (only over completion, not prompt)
        log_probs = log_probs.sum(dim=1) / mask.float().sum(dim=1).clamp(min=1)
    else:
        # Average over sequence
        log_probs = log_probs.mean(dim=1)
    
    return log_probs


def compute_dpo_loss(policy_log_probs_w, policy_log_probs_l, ref_log_probs_w, ref_log_probs_l, beta=0.1):
    """
    Compute DPO loss.
    
    Loss = -log(sigmoid(beta * (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))))
    
    Args:
        policy_log_probs_w: Log probs under policy for preferred completion
        policy_log_probs_l: Log probs under policy for less preferred completion  
        ref_log_probs_w: Log probs under reference for preferred completion
        ref_log_probs_l: Log probs under reference for less preferred completion
        beta: Temperature parameter (higher = stay closer to reference)
    
    Returns:
        loss: Scalar loss
        accuracy: Accuracy (fraction where policy prefers winner)
    """
    # Compute logits for DPO
    policy_diff = policy_log_probs_w - policy_log_probs_l
    ref_diff = ref_log_probs_w - ref_log_probs_l
    
    logits = beta * (policy_diff - ref_diff)
    
    # Loss: -log(sigmoid(logits))
    loss = -F.logsigmoid(logits).mean()
    
    # Accuracy: how often does policy prefer winner?
    accuracy = (policy_diff > 0).float().mean()
    
    # Also compute implicit reward (for monitoring)
    implicit_reward_w = beta * (policy_log_probs_w - ref_log_probs_w)
    implicit_reward_l = beta * (policy_log_probs_l - ref_log_probs_l)
    
    return loss, accuracy, implicit_reward_w.mean(), implicit_reward_l.mean()


def train_epoch(policy_model, ref_model, dataloader, optimizer, device, beta=0.1, block_size=256):
    """Train for one epoch."""
    policy_model.train()
    ref_model.eval()
    
    total_loss = 0
    total_accuracy = 0
    total_reward_w = 0
    total_reward_l = 0
    
    for batch in dataloader:
        tokens_w = batch['tokens_w'].to(device)
        tokens_l = batch['tokens_l'].to(device)
        prompt_lens = batch['prompt_lens'].to(device)
        
        # Get log probs under policy
        policy_log_probs_w = get_log_probs(policy_model, tokens_w, prompt_lens)
        policy_log_probs_l = get_log_probs(policy_model, tokens_l, prompt_lens)
        
        # Get log probs under reference (no grad)
        with torch.no_grad():
            ref_log_probs_w = get_log_probs(ref_model, tokens_w, prompt_lens)
            ref_log_probs_l = get_log_probs(ref_model, tokens_l, prompt_lens)
        
        # Compute DPO loss
        loss, accuracy, reward_w, reward_l = compute_dpo_loss(
            policy_log_probs_w, policy_log_probs_l,
            ref_log_probs_w, ref_log_probs_l,
            beta=beta
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_accuracy += accuracy.item()
        total_reward_w += reward_w.item()
        total_reward_l += reward_l.item()
    
    n = len(dataloader)
    return total_loss/n, total_accuracy/n, total_reward_w/n, total_reward_l/n


@torch.no_grad()
def eval_epoch(policy_model, ref_model, dataloader, device, beta=0.1, block_size=256):
    """Evaluate for one epoch."""
    policy_model.eval()
    ref_model.eval()
    
    total_loss = 0
    total_accuracy = 0
    total_reward_w = 0
    total_reward_l = 0
    
    for batch in dataloader:
        tokens_w = batch['tokens_w'].to(device)
        tokens_l = batch['tokens_l'].to(device)
        prompt_lens = batch['prompt_lens'].to(device)
        
        policy_log_probs_w = get_log_probs(policy_model, tokens_w, prompt_lens)
        policy_log_probs_l = get_log_probs(policy_model, tokens_l, prompt_lens)
        
        ref_log_probs_w = get_log_probs(ref_model, tokens_w, prompt_lens)
        ref_log_probs_l = get_log_probs(ref_model, tokens_l, prompt_lens)
        
        loss, accuracy, reward_w, reward_l = compute_dpo_loss(
            policy_log_probs_w, policy_log_probs_l,
            ref_log_probs_w, ref_log_probs_l,
            beta=beta
        )
        
        total_loss += loss.item()
        total_accuracy += accuracy.item()
        total_reward_w += reward_w.item()
        total_reward_l += reward_l.item()
    
    n = len(dataloader)
    return total_loss/n, total_accuracy/n, total_reward_w/n, total_reward_l/n


def prepare_dpo_data(preference_data_path: str, encode_fn, output_path: str):
    """
    Convert preference pairs to DPO format.
    
    Converts full_a/full_b to tokens_w/tokens_l based on preference.
    """
    print(f"Preparing DPO data from {preference_data_path}...")
    
    with open(preference_data_path, 'rb') as f:
        data = pickle.load(f)
    
    dpo_data = []
    for item in data:
        # Encode texts
        if isinstance(item['full_a'], str):
            tokens_a = encode_fn(item['full_a'])
            tokens_b = encode_fn(item['full_b'])
            prompt = encode_fn(item['prompt'])
            prompt_len = len(prompt)
        else:
            tokens_a = item['full_a']
            tokens_b = item['full_b']
            prompt_len = item.get('prompt_len', 64)
        
        # Assign winner/loser based on preference
        if item['preference'] == 0:  # A preferred
            tokens_w = tokens_a
            tokens_l = tokens_b
        else:  # B preferred
            tokens_w = tokens_b
            tokens_l = tokens_a
        
        dpo_data.append({
            'tokens_w': tokens_w,
            'tokens_l': tokens_l,
            'prompt_len': prompt_len
        })
    
    with open(output_path, 'wb') as f:
        pickle.dump(dpo_data, f)
    
    print(f"✓ Saved {len(dpo_data)} DPO pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train with DPO')
    parser.add_argument('--ref_checkpoint', type=str, default='out-shakespeare/ckpt.pt',
                       help='Reference model checkpoint (frozen)')
    parser.add_argument('--init_checkpoint', type=str, default=None,
                       help='Initial policy checkpoint (default: same as ref)')
    parser.add_argument('--train_data', type=str, default='data/preferences/train.pkl',
                       help='Training preference data')
    parser.add_argument('--val_data', type=str, default='data/preferences/val.pkl',
                       help='Validation preference data')
    parser.add_argument('--out_dir', type=str, default='out-dpo',
                       help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--block_size', type=int, default=256,
                       help='Block size')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-6,
                       help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.1,
                       help='DPO temperature (higher = stay closer to reference)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print("DPO Training")
    print(f"{'='*60}\n")
    
    # Load reference model (frozen)
    print("Loading reference model...")
    checkpoint = torch.load(args.ref_checkpoint, map_location=args.device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    ref_model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    ref_model.load_state_dict(state_dict)
    ref_model.to(args.device)
    ref_model.eval()
    
    for param in ref_model.parameters():
        param.requires_grad = False
    
    print("✓ Reference model loaded (frozen)")
    
    # Load/create policy model
    if args.init_checkpoint is None:
        args.init_checkpoint = args.ref_checkpoint
    
    print(f"Loading policy model from {args.init_checkpoint}...")
    checkpoint = torch.load(args.init_checkpoint, map_location=args.device)
    policy_model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    policy_model.load_state_dict(state_dict)
    policy_model.to(args.device)
    
    print("✓ Policy model loaded")
    
    # Get encoder function
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:
        dataset = checkpoint['config']['dataset']
        meta_path = os.path.join('data', dataset, 'meta.pkl')
        if os.path.exists(meta_path):
            import pickle
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi = meta['stoi']
            encode = lambda s: [stoi[c] for c in s]
        else:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s)
    else:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s)
    
    # Prepare DPO datasets
    train_dpo_path = args.train_data.replace('.pkl', '_dpo.pkl')
    val_dpo_path = args.val_data.replace('.pkl', '_dpo.pkl')
    
    if not os.path.exists(train_dpo_path):
        prepare_dpo_data(args.train_data, encode, train_dpo_path)
    if not os.path.exists(val_dpo_path):
        prepare_dpo_data(args.val_data, encode, val_dpo_path)
    
    # Load datasets
    train_dataset = PreferenceDataset(train_dpo_path)
    val_dataset = PreferenceDataset(val_dpo_path)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, args.block_size)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, args.block_size)
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)
    
    print(f"\nStarting DPO training...")
    print(f"Beta: {args.beta}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs}\n")
    
    best_val_acc = 0
    
    for epoch in range(args.num_epochs):
        t0 = time.time()
        
        train_loss, train_acc, train_rw_w, train_rw_l = train_epoch(
            policy_model, ref_model, train_loader, optimizer, args.device, args.beta, args.block_size
        )
        
        val_loss, val_acc, val_rw_w, val_rw_l = eval_epoch(
            policy_model, ref_model, val_loader, args.device, args.beta, args.block_size
        )
        
        t1 = time.time()
        
        print(f"Epoch {epoch+1}/{args.num_epochs} ({t1-t0:.2f}s):")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, R_w: {train_rw_w:.4f}, R_l: {train_rw_l:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, R_w: {val_rw_w:.4f}, R_l: {val_rw_l:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'model': policy_model.state_dict(),
                'model_args': vars(gptconf),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt.pt'))
            print(f"  → Saved best model")
        print()
    
    print(f"{'='*60}")
    print("DPO Training Complete!")
    print(f"{'='*60}")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {args.out_dir}/ckpt.pt")


if __name__ == '__main__':
    main()

