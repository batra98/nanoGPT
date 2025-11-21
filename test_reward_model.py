"""
Test the trained reward model on generated samples.

Shows high-reward and low-reward samples to validate the reward model
is learning the dialogue density preference.
"""

import os
import pickle
import argparse
import torch
import numpy as np
from contextlib import nullcontext

from model import GPTConfig, GPT
from reward_model import RewardModel
from preference_heuristic import compute_dialogue_density, compute_dialogue_stats


def load_reward_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained reward model from checkpoint."""
    print(f"Loading reward model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get GPT checkpoint path from args
    gpt_checkpoint_path = checkpoint['args']['gpt_checkpoint']
    
    # Load GPT
    gpt_checkpoint = torch.load(gpt_checkpoint_path, map_location=device)
    gptconf = GPTConfig(**gpt_checkpoint['model_args'])
    gpt = GPT(gptconf)
    
    state_dict = gpt_checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    gpt.load_state_dict(state_dict)
    gpt.to(device)
    
    # Create reward model
    reward_model = RewardModel(gpt, freeze_gpt=True)
    
    # Load reward model state
    reward_model.load_state_dict(checkpoint['model_state_dict'])
    reward_model.to(device)
    reward_model.eval()
    
    print(f"✓ Reward model loaded")
    print(f"  Training accuracy: {checkpoint['train_acc']:.4f}")
    print(f"  Validation accuracy: {checkpoint['val_acc']:.4f}")
    
    return reward_model


def load_gpt_model(checkpoint_path: str, device: str = 'cuda'):
    """Load GPT model for generation."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    return model


def get_encoder_decoder(checkpoint_path: str):
    """Get encoder/decoder functions."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:
        dataset = checkpoint['config']['dataset']
        meta_path = os.path.join('data', dataset, 'meta.pkl')
        
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])
            return encode, decode
    
    # Fall back to tiktoken
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    return encode, decode


def generate_samples(model, encode, decode, num_samples=50, max_new_tokens=200, device='cuda'):
    """Generate samples from the model."""
    samples = []
    
    # Simple prompts
    prompts = [
        "\n",
        "ROMEO:\n",
        "First Citizen:\n",
        "KING:\n",
        "The ",
    ] * (num_samples // 5 + 1)
    
    print(f"\nGenerating {num_samples} samples...")
    
    for i in range(num_samples):
        prompt = prompts[i]
        start_ids = encode(prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
        
        with torch.no_grad():
            y = model.generate(x, max_new_tokens, temperature=0.8, top_k=200)
        
        text = decode(y[0].tolist())
        samples.append(text)
    
    return samples


def main():
    parser = argparse.ArgumentParser(description='Test reward model')
    parser.add_argument('--reward_checkpoint', type=str, default='out-reward-model/best_model.pt',
                       help='Path to reward model checkpoint')
    parser.add_argument('--gpt_checkpoint', type=str, default='out-shakespeare/ckpt.pt',
                       help='Path to GPT checkpoint for generation')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to generate and evaluate')
    parser.add_argument('--max_new_tokens', type=int, default=200,
                       help='Max tokens to generate per sample')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--output_file', type=str, default='results/reward_model_test.txt',
                       help='Output file for results')
    parser.add_argument('--wandb_log', action='store_true', default=False,
                       help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='rlhf-reward-model',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='reward-test',
                       help='Wandb run name')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    print(f"{'='*60}")
    print("Testing Reward Model")
    print(f"{'='*60}\n")
    
    # Load models
    reward_model = load_reward_model(args.reward_checkpoint, args.device)
    gpt_model = load_gpt_model(args.gpt_checkpoint, args.device)
    encode, decode = get_encoder_decoder(args.gpt_checkpoint)
    
    # Generate samples
    samples = generate_samples(
        gpt_model, encode, decode,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        device=args.device
    )
    
    # Compute rewards and dialogue densities
    print("\nComputing rewards...")
    rewards = []
    densities = []
    
    for sample in samples:
        # Encode sample
        tokens = encode(sample)
        tokens = tokens[:256]  # Truncate to block size
        
        # Pad if needed
        if len(tokens) < 256:
            tokens = tokens + [0] * (256 - len(tokens))
        
        tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=args.device)[None, ...]
        
        # Get reward
        with torch.no_grad():
            reward = reward_model(tokens_tensor).item()
        
        # Get actual dialogue density
        density = compute_dialogue_density(sample)
        
        rewards.append(reward)
        densities.append(density)
    
    # Sort by reward
    sorted_indices = np.argsort(rewards)
    
    # Statistics
    print(f"\n{'='*60}")
    print("Reward Statistics")
    print(f"{'='*60}")
    print(f"Mean reward: {np.mean(rewards):.4f}")
    print(f"Std reward: {np.std(rewards):.4f}")
    print(f"Min reward: {np.min(rewards):.4f}")
    print(f"Max reward: {np.max(rewards):.4f}")
    print()
    print(f"Mean dialogue density: {np.mean(densities):.4f}")
    print(f"Std dialogue density: {np.std(densities):.4f}")
    
    # Correlation
    correlation = np.corrcoef(rewards, densities)[0, 1]
    print(f"\nCorrelation (reward vs density): {correlation:.4f}")
    
    # Initialize wandb and log metrics
    if args.wandb_log:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                'reward_checkpoint': args.reward_checkpoint,
                'gpt_checkpoint': args.gpt_checkpoint,
                'num_samples': args.num_samples,
            }
        )
        
        # Log statistics
        wandb.log({
            'test/mean_reward': np.mean(rewards),
            'test/std_reward': np.std(rewards),
            'test/min_reward': np.min(rewards),
            'test/max_reward': np.max(rewards),
            'test/mean_density': np.mean(densities),
            'test/std_density': np.std(densities),
            'test/correlation': correlation,
        })
        
        # Log reward distribution
        wandb.log({
            'reward_distribution': wandb.Histogram(rewards),
            'density_distribution': wandb.Histogram(densities),
        })
        
        # Log scatter plot
        wandb.log({
            'reward_vs_density': wandb.Scatter(
                x=densities,
                y=rewards,
                xname='Dialogue Density',
                yname='Reward'
            )
        })
    
    # Save results to file
    with open(args.output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Reward Model Test Results\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Reward model: {args.reward_checkpoint}\n")
        f.write(f"GPT model: {args.gpt_checkpoint}\n")
        f.write(f"Number of samples: {args.num_samples}\n\n")
        
        f.write("Statistics:\n")
        f.write(f"  Mean reward: {np.mean(rewards):.4f}\n")
        f.write(f"  Mean dialogue density: {np.mean(densities):.4f}\n")
        f.write(f"  Correlation: {correlation:.4f}\n\n")
        
        f.write("="*60 + "\n")
        f.write("TOP 10 HIGHEST REWARD SAMPLES\n")
        f.write("="*60 + "\n\n")
        
        for i in range(min(10, len(samples))):
            idx = sorted_indices[-(i+1)]
            f.write(f"--- Sample {i+1} (Reward: {rewards[idx]:.4f}, Density: {densities[idx]:.4f}) ---\n")
            f.write(samples[idx][:500])
            f.write("\n\n")
        
        f.write("="*60 + "\n")
        f.write("TOP 10 LOWEST REWARD SAMPLES\n")
        f.write("="*60 + "\n\n")
        
        for i in range(min(10, len(samples))):
            idx = sorted_indices[i]
            f.write(f"--- Sample {i+1} (Reward: {rewards[idx]:.4f}, Density: {densities[idx]:.4f}) ---\n")
            f.write(samples[idx][:500])
            f.write("\n\n")
    
    print(f"\n✓ Results saved to {args.output_file}")
    
    # Print some examples to console
    print(f"\n{'='*60}")
    print("Example HIGH REWARD Sample")
    print(f"{'='*60}")
    idx = sorted_indices[-1]
    print(f"Reward: {rewards[idx]:.4f}, Density: {densities[idx]:.4f}")
    print(samples[idx][:300])
    
    print(f"\n{'='*60}")
    print("Example LOW REWARD Sample")
    print(f"{'='*60}")
    idx = sorted_indices[0]
    print(f"Reward: {rewards[idx]:.4f}, Density: {densities[idx]:.4f}")
    print(samples[idx][:300])
    
    if args.wandb_log:
        wandb.finish()


if __name__ == '__main__':
    main()

