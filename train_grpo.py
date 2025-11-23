"""
Train model using GRPO (Group Relative Policy Optimization) with RLVR.

GRPO objective:
- Sample completions from policy: y ~ π_θ(·|x)
- Compute verifier reward: r = v(y)
- Compute importance weight: w = π_θ(y|x) / π_ref(y|x)
- Loss: L = -E[r * w] + β * KL(π_θ || π_ref)

Paper reference: GRPO / RLVR
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import numpy as np
from tqdm import tqdm

from model import GPTConfig, GPT
from verifier import compute_verifier_score


class PromptDataset(Dataset):
    """Dataset for prompts."""
    
    def __init__(self, prompts_path: str):
        with open(prompts_path, 'rb') as f:
            self.prompts = pickle.load(f)
        print(f"Loaded {len(self.prompts)} prompts from {prompts_path}")
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return torch.tensor(self.prompts[idx], dtype=torch.long)


def collate_prompts(batch, block_size=256):
    """Collate prompts with padding."""
    max_len = min(max(len(p) for p in batch), block_size)
    padded = []
    for p in batch:
        if len(p) > max_len:
            p = p[:max_len]
        else:
            p = torch.cat([p, torch.zeros(max_len - len(p), dtype=torch.long)])
        padded.append(p)
    return torch.stack(padded)


def get_log_probs(model, tokens, prompt_len=None):
    """
    Get log probabilities of tokens under the model.
    
    Args:
        model: GPT model (may be wrapped in DDP)
        tokens: Token IDs (batch_size, seq_len)
        prompt_len: Length of prompt (to only compute on completion)
    
    Returns:
        log_probs: Log probabilities (batch_size,)
    """
    # Get the actual model (unwrap DDP if needed)
    if hasattr(model, 'module'):
        raw_model = model.module
    else:
        raw_model = model
    
    input_tokens = tokens[:, :-1]  # All but last token
    target_tokens = tokens[:, 1:]   # All but first token (shifted)
    
    # Manually forward through model to get full logits without loss computation
    device = input_tokens.device
    b, t = input_tokens.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device)
    
    # Forward through transformer
    tok_emb = raw_model.transformer.wte(input_tokens)
    pos_emb = raw_model.transformer.wpe(pos)
    x = raw_model.transformer.drop(tok_emb + pos_emb)
    for block in raw_model.transformer.h:
        x = block(x)
    x = raw_model.transformer.ln_f(x)
    
    # Get logits for all positions
    logits = raw_model.lm_head(x)  # (batch_size, seq_len, vocab_size)
    
    # Get log probs
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Gather log probs of actual next tokens
    log_probs = torch.gather(log_probs, -1, target_tokens.unsqueeze(-1)).squeeze(-1)
    # Now log_probs shape: (batch_size, seq_len)
    
    # Mask out prompt tokens if prompt_len is provided
    if prompt_len is not None:
        seq_len = log_probs.size(1)
        mask = torch.arange(seq_len, device=log_probs.device)[None, :] >= prompt_len[:, None]
        log_probs = log_probs * mask.float()
        # Sum log probs (only over completion, not prompt)
        log_probs = log_probs.sum(dim=1) / mask.float().sum(dim=1).clamp(min=1)
    else:
        # Average over sequence
        log_probs = log_probs.mean(dim=1)
    
    return log_probs


def sample_completions(model, prompts, max_new_tokens=200, temperature=0.8, top_k=200, num_samples_per_prompt=4):
    """
    Sample completions from model.
    
    Args:
        model: GPT model
        prompts: Prompt tensors (batch_size, prompt_len)
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        num_samples_per_prompt: Number of samples per prompt
    
    Returns:
        completions: List of completion tensors
        prompt_lens: Prompt lengths
    """
    model.eval()
    completions = []
    prompt_lens = []
    
    with torch.no_grad():
        for prompt in prompts:
            prompt_len = prompt.size(0)
            prompt_lens.append(prompt_len)
            
            # Expand prompt for multiple samples
            prompt_expanded = prompt.unsqueeze(0).repeat(num_samples_per_prompt, 1)
            
            # Generate
            completion = model.generate(
                prompt_expanded,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
            
            completions.append(completion)
    
    return completions, prompt_lens


def compute_grpo_loss(policy_log_probs, ref_log_probs, rewards, beta=0.1):
    """
    Compute GRPO loss with importance sampling.
    
    Loss = -E[r * w] + β * KL(π_θ || π_ref)
    where w = π_θ(y|x) / π_ref(y|x) is the importance weight
    
    Args:
        policy_log_probs: Log probs under policy (batch_size,)
        ref_log_probs: Log probs under reference (batch_size,)
        rewards: Verifier rewards (batch_size,)
        beta: KL penalty weight
    
    Returns:
        loss: Scalar loss
        mean_reward: Mean reward
        mean_importance_weight: Mean importance weight
        kl_divergence: KL divergence
    """
    # Importance weights: w = exp(log π_θ - log π_ref) = π_θ / π_ref
    log_importance_weights = policy_log_probs - ref_log_probs
    importance_weights = torch.exp(log_importance_weights)
    
    # Weighted reward: r * w
    weighted_rewards = rewards * importance_weights
    
    # Policy gradient loss: -E[r * w] (negative because we maximize)
    policy_loss = -weighted_rewards.mean()
    
    # KL divergence: KL(π_θ || π_ref) = E[log π_θ - log π_ref]
    kl_div = (policy_log_probs - ref_log_probs).mean()
    
    # Total loss
    loss = policy_loss + beta * kl_div
    
    return loss, weighted_rewards.mean(), importance_weights.mean(), kl_div


def train_step(policy_model, ref_model, prompts, encode, decode, device, max_new_tokens=200, 
               temperature=0.8, top_k=200, num_samples_per_prompt=4, beta=0.1, block_size=256):
    """Single training step."""
    policy_model.train()
    ref_model.eval()
    
    # Sample completions from policy
    completions_list, prompt_lens = sample_completions(
        policy_model, prompts, max_new_tokens, temperature, top_k, num_samples_per_prompt
    )
    
    # Flatten completions
    all_completions = []
    all_prompt_lens = []
    for completions, prompt_len in zip(completions_list, prompt_lens):
        for comp in completions:
            all_completions.append(comp)
            all_prompt_lens.append(prompt_len)
    
    if len(all_completions) == 0:
        return None, None, None, None
    
    # Pad/truncate to block_size
    max_len = min(max(c.size(0) for c in all_completions), block_size)
    padded_completions = []
    for comp in all_completions:
        if comp.size(0) > max_len:
            comp = comp[:max_len]
        else:
            comp = torch.cat([comp, torch.zeros(max_len - comp.size(0), dtype=torch.long, device=comp.device)])
        padded_completions.append(comp)
    
    completions_tensor = torch.stack(padded_completions).to(device)
    prompt_lens_tensor = torch.tensor(all_prompt_lens, device=device)
    
    # Compute verifier rewards
    rewards = []
    for comp in all_completions:
        text = decode(comp.cpu().tolist())
        reward = compute_verifier_score(text)
        rewards.append(reward)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    
    # Get log probs under policy
    policy_log_probs = get_log_probs(policy_model, completions_tensor, prompt_lens_tensor)
    
    # Get log probs under reference (no grad)
    with torch.no_grad():
        ref_log_probs = get_log_probs(ref_model, completions_tensor, prompt_lens_tensor)
    
    # Compute GRPO loss
    loss, mean_weighted_reward, mean_importance_weight, kl_div = compute_grpo_loss(
        policy_log_probs, ref_log_probs, rewards, beta
    )
    
    return loss, mean_weighted_reward.item(), mean_importance_weight.item(), kl_div.item(), rewards.mean().item()


def main():
    parser = argparse.ArgumentParser(description='Train with GRPO (RLVR)')
    parser.add_argument('--ref_checkpoint', type=str, default='out-shakespeare/ckpt.pt',
                       help='Reference model checkpoint (frozen)')
    parser.add_argument('--init_checkpoint', type=str, default=None,
                       help='Initial policy checkpoint (default: same as ref)')
    parser.add_argument('--prompts_path', type=str, default='data/grpo/prompts.pkl',
                       help='Path to prompts pickle file')
    parser.add_argument('--out_dir', type=str, default='out-grpo',
                       help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (number of prompts per step)')
    parser.add_argument('--block_size', type=int, default=256,
                       help='Block size')
    parser.add_argument('--num_steps', type=int, default=1000,
                       help='Number of training steps')
    parser.add_argument('--num_samples_per_prompt', type=int, default=4,
                       help='Number of samples per prompt')
    parser.add_argument('--max_new_tokens', type=int, default=200,
                       help='Max new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=200,
                       help='Top-k sampling')
    parser.add_argument('--learning_rate', type=float, default=1e-6,
                       help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.1,
                       help='KL penalty weight')
    parser.add_argument('--eval_interval', type=int, default=100,
                       help='Evaluation interval')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    parser.add_argument('--wandb_log', action='store_true', default=True,
                       help='Enable wandb logging (default: True)')
    parser.add_argument('--wandb_project', type=str, default='grpo-rlvr',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name')
    
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
    else:
        master_process = True
        ddp_world_size = 1
    
    if master_process:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"{'='*60}")
        print("GRPO Training")
        print(f"{'='*60}\n")
    
    # Load reference model
    if master_process:
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
    
    if master_process:
        print("✓ Reference model loaded (frozen)")
    
    # Load/create policy model
    if args.init_checkpoint is None:
        args.init_checkpoint = args.ref_checkpoint
    
    if master_process:
        print(f"Loading policy model from {args.init_checkpoint}...")
    checkpoint = torch.load(args.init_checkpoint, map_location=args.device)
    policy_model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    policy_model.load_state_dict(state_dict)
    policy_model.to(args.device)
    
    # Wrap policy model in DDP
    if ddp:
        policy_model = DDP(policy_model, device_ids=[ddp_local_rank])
        raw_policy_model = policy_model.module
    else:
        raw_policy_model = policy_model
    
    if master_process:
        print("✓ Policy model loaded")
    
    # Get encode/decode
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:
        dataset = checkpoint['config']['dataset']
        meta_path = os.path.join('data', dataset, 'meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi = meta['stoi']
            itos = meta['itos']
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])
        else:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            decode = lambda l: enc.decode(l)
    else:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    
    # Load prompts
    if master_process:
        print(f"\nLoading prompts from {args.prompts_path}...")
    dataset = PromptDataset(args.prompts_path)
    
    # Create dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if ddp else None
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=lambda b: collate_prompts(b, args.block_size)
    )
    
    if master_process:
        print(f"Total prompts: {len(dataset)}")
        print(f"Batch size: {args.batch_size} prompts per step")
        print(f"Samples per prompt: {args.num_samples_per_prompt}")
        print(f"Total samples per step: {args.batch_size * args.num_samples_per_prompt}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)
    
    # Initialize wandb
    if args.wandb_log and master_process:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"grpo-{int(time.time())}",
            config={
                'ref_checkpoint': args.ref_checkpoint,
                'batch_size': args.batch_size,
                'num_samples_per_prompt': args.num_samples_per_prompt,
                'block_size': args.block_size,
                'num_steps': args.num_steps,
                'learning_rate': args.learning_rate,
                'beta': args.beta,
                'max_new_tokens': args.max_new_tokens,
                'temperature': args.temperature,
                'ddp_world_size': ddp_world_size,
            }
        )
    
    if master_process:
        print(f"\nStarting GRPO training...")
        print(f"Beta (KL weight): {args.beta}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Training steps: {args.num_steps}")
        print(f"DDP: {ddp}, World size: {ddp_world_size}\n")
    
    # Training loop
    step = 0
    best_mean_reward = -float('inf')
    reward_history = []
    
    # Create iterator that cycles through data
    data_iter = iter(train_loader)
    
    if master_process:
        print(f"{'='*60}")
        print("Starting Training")
        print(f"{'='*60}\n")
    
    while step < args.num_steps:
        # Get next batch
        try:
            prompts = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            prompts = next(data_iter)
        
        prompts = prompts.to(args.device)
        
        # Training step
        result = train_step(
            policy_model, ref_model, prompts, encode, decode, args.device,
            args.max_new_tokens, args.temperature, args.top_k,
            args.num_samples_per_prompt, args.beta, args.block_size
        )
        
        if result is None:
            continue
        
        loss, mean_weighted_reward, mean_importance_weight, kl_div, mean_reward = result
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()
        
        reward_history.append(mean_reward)
        
        if master_process:
            if step % 10 == 0:
                print(f"Step {step}/{args.num_steps}: "
                      f"Loss={loss.item():.4f}, "
                      f"Reward={mean_reward:.2f}, "
                      f"Weighted Reward={mean_weighted_reward:.4f}, "
                      f"KL={kl_div:.4f}")
            
            # Log to wandb
            if args.wandb_log:
                wandb.log({
                    'step': step,
                    'loss': loss.item(),
                    'reward': mean_reward,
                    'weighted_reward': mean_weighted_reward,
                    'importance_weight': mean_importance_weight,
                    'kl_divergence': kl_div,
                })
            
            # Evaluation
            if step % args.eval_interval == 0 and step > 0:
                # Compute mean reward over recent history
                recent_rewards = reward_history[-100:] if len(reward_history) >= 100 else reward_history
                avg_reward = np.mean(recent_rewards)
                
                print(f"\nEvaluation at step {step}:")
                print(f"  Mean reward (last 100 steps): {avg_reward:.2f}")
                
                if avg_reward > best_mean_reward:
                    best_mean_reward = avg_reward
                    checkpoint = {
                        'model': raw_policy_model.state_dict(),
                        'model_args': vars(gptconf),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'mean_reward': avg_reward,
                        'args': vars(args)
                    }
                    torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt.pt'))
                    print(f"  → Saved best model (mean_reward={avg_reward:.2f})")
                
                if args.wandb_log:
                    wandb.log({
                        'eval/mean_reward': avg_reward,
                        'eval/best_mean_reward': best_mean_reward,
                    })
                print()
        
        step += 1
    
    if master_process:
        print(f"{'='*60}")
        print("GRPO Training Complete!")
        print(f"{'='*60}")
        print(f"Best mean reward: {best_mean_reward:.2f}")
        print(f"Model saved to: {args.out_dir}/ckpt.pt")
        
        # Save reward history
        reward_history_path = os.path.join(args.out_dir, 'reward_history.pkl')
        with open(reward_history_path, 'wb') as f:
            pickle.dump(reward_history, f)
        print(f"Reward history saved to: {reward_history_path}")
        
        if args.wandb_log:
            wandb.log({'final/best_mean_reward': best_mean_reward})
            wandb.finish()
    
    if ddp:
        destroy_process_group()


if __name__ == '__main__':
    main()

