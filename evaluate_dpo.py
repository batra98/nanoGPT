"""
Evaluate DPO-aligned model vs base model.

Compares:
- Sample quality (dialogue density)
- Reward model scores
- KL divergence from reference
"""

import os
import pickle
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import GPTConfig, GPT
from reward_model import RewardModel, create_reward_model
from preference_heuristic import compute_dialogue_density


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load GPT model."""
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
    """Get encoder/decoder."""
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
    
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    return encode, decode


def generate_samples(model, encode, decode, num_samples=50, max_new_tokens=200, device='cuda', seed=42):
    """Generate samples."""
    torch.manual_seed(seed)
    samples = []
    
    prompts = [
        "\n",
        "ROMEO:\n",
        "First Citizen:\n",
        "JULIET:\n",
        "The ",
    ] * (num_samples // 5 + 1)
    
    print(f"Generating {num_samples} samples...")
    
    for i in range(num_samples):
        prompt = prompts[i]
        start_ids = encode(prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
        
        with torch.no_grad():
            y = model.generate(x, max_new_tokens, temperature=0.8, top_k=200)
        
        text = decode(y[0].tolist())
        samples.append(text)
    
    return samples


def evaluate_samples(samples, reward_model, encode, device='cuda'):
    """Evaluate samples with reward model and dialogue density."""
    rewards = []
    densities = []
    
    for sample in samples:
        # Get reward
        tokens = encode(sample)[:256]
        if len(tokens) < 256:
            tokens = tokens + [0] * (256 - len(tokens))
        tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device)[None, ...]
        
        with torch.no_grad():
            reward = reward_model(tokens_tensor).item()
        
        # Get dialogue density
        density = compute_dialogue_density(sample)
        
        rewards.append(reward)
        densities.append(density)
    
    return rewards, densities


def compute_kl_divergence(base_model, dpo_model, encode, val_data_path, device='cuda', num_samples=100):
    """Compute KL divergence between DPO and base model."""
    import numpy as np
    
    # Load validation data
    val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')
    
    kl_divs = []
    
    for _ in range(num_samples):
        # Random sequence
        start_idx = np.random.randint(0, len(val_data) - 257)
        tokens = torch.tensor(val_data[start_idx:start_idx+256].astype(np.int64), dtype=torch.long, device=device)[None, ...]
        
        with torch.no_grad():
            # Get logits from both models
            base_logits, _ = base_model(tokens[:, :-1])
            dpo_logits, _ = dpo_model(tokens[:, :-1])
            
            # Compute log probs
            base_log_probs = torch.log_softmax(base_logits, dim=-1)
            dpo_log_probs = torch.log_softmax(dpo_logits, dim=-1)
            
            # KL divergence: sum over vocab, average over sequence
            kl = (torch.exp(dpo_log_probs) * (dpo_log_probs - base_log_probs)).sum(dim=-1).mean()
            kl_divs.append(kl.item())
    
    return np.mean(kl_divs), np.std(kl_divs)


def main():
    parser = argparse.ArgumentParser(description='Evaluate DPO model')
    parser.add_argument('--base_checkpoint', type=str, default='out-shakespeare/ckpt.pt',
                       help='Base model checkpoint')
    parser.add_argument('--dpo_checkpoint', type=str, default='out-dpo/ckpt.pt',
                       help='DPO model checkpoint')
    parser.add_argument('--reward_checkpoint', type=str, default='out-reward-model/best_model.pt',
                       help='Reward model checkpoint')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to generate')
    parser.add_argument('--max_new_tokens', type=int, default=200,
                       help='Max tokens to generate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print("Evaluating DPO Model")
    print(f"{'='*60}\n")
    
    # Load models
    print("Loading models...")
    base_model = load_model(args.base_checkpoint, args.device)
    dpo_model = load_model(args.dpo_checkpoint, args.device)
    
    # Load reward model
    reward_checkpoint = torch.load(args.reward_checkpoint, map_location=args.device)
    gpt_ckpt_path = reward_checkpoint['args']['gpt_checkpoint']
    reward_model = create_reward_model(gpt_ckpt_path, args.device, freeze_gpt=True)
    reward_model.load_state_dict(reward_checkpoint['model_state_dict'])
    reward_model.eval()
    
    encode, decode = get_encoder_decoder(args.base_checkpoint)
    
    print("✓ Models loaded\n")
    
    # Generate samples from both models
    print("Generating samples from BASE model...")
    base_samples = generate_samples(base_model, encode, decode, args.num_samples, args.max_new_tokens, args.device, seed=42)
    
    print("Generating samples from DPO model...")
    dpo_samples = generate_samples(dpo_model, encode, decode, args.num_samples, args.max_new_tokens, args.device, seed=42)
    
    # Evaluate samples
    print("\nEvaluating BASE samples...")
    base_rewards, base_densities = evaluate_samples(base_samples, reward_model, encode, args.device)
    
    print("Evaluating DPO samples...")
    dpo_rewards, dpo_densities = evaluate_samples(dpo_samples, reward_model, encode, args.device)
    
    # Compute KL divergence
    print("\nComputing KL divergence...")
    val_data_path = 'data/shakespeare/val.bin'
    if os.path.exists(val_data_path):
        kl_mean, kl_std = compute_kl_divergence(base_model, dpo_model, encode, val_data_path, args.device)
        print(f"KL divergence: {kl_mean:.4f} ± {kl_std:.4f}")
    else:
        kl_mean, kl_std = None, None
        print("Val data not found, skipping KL divergence")
    
    # Statistics
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}\n")
    
    print("Reward Model Scores:")
    print(f"  Base model: {np.mean(base_rewards):.4f} ± {np.std(base_rewards):.4f}")
    print(f"  DPO model:  {np.mean(dpo_rewards):.4f} ± {np.std(dpo_rewards):.4f}")
    print(f"  Improvement: {np.mean(dpo_rewards) - np.mean(base_rewards):.4f}")
    
    print("\nDialogue Density:")
    print(f"  Base model: {np.mean(base_densities):.4f} ± {np.std(base_densities):.4f}")
    print(f"  DPO model:  {np.mean(dpo_densities):.4f} ± {np.std(dpo_densities):.4f}")
    print(f"  Improvement: {np.mean(dpo_densities) - np.mean(base_densities):.4f}")
    
    if kl_mean is not None:
        print(f"\nKL Divergence: {kl_mean:.4f} ± {kl_std:.4f}")
    
    # Save results
    output_file = os.path.join(args.output_dir, 'dpo_evaluation.txt')
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DPO Evaluation Results\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Base model: {args.base_checkpoint}\n")
        f.write(f"DPO model: {args.dpo_checkpoint}\n")
        f.write(f"Reward model: {args.reward_checkpoint}\n\n")
        
        f.write("Reward Scores:\n")
        f.write(f"  Base: {np.mean(base_rewards):.4f} ± {np.std(base_rewards):.4f}\n")
        f.write(f"  DPO:  {np.mean(dpo_rewards):.4f} ± {np.std(dpo_rewards):.4f}\n")
        f.write(f"  Improvement: {np.mean(dpo_rewards) - np.mean(base_rewards):.4f}\n\n")
        
        f.write("Dialogue Density:\n")
        f.write(f"  Base: {np.mean(base_densities):.4f} ± {np.std(base_densities):.4f}\n")
        f.write(f"  DPO:  {np.mean(dpo_densities):.4f} ± {np.std(dpo_densities):.4f}\n")
        f.write(f"  Improvement: {np.mean(dpo_densities) - np.mean(base_densities):.4f}\n\n")
        
        if kl_mean is not None:
            f.write(f"KL Divergence: {kl_mean:.4f} ± {kl_std:.4f}\n\n")
        
        f.write("="*60 + "\n")
        f.write("Example BASE Samples\n")
        f.write("="*60 + "\n\n")
        for i in range(min(5, len(base_samples))):
            f.write(f"--- Sample {i+1} (Reward: {base_rewards[i]:.4f}, Density: {base_densities[i]:.4f}) ---\n")
            f.write(base_samples[i][:300] + "\n\n")
        
        f.write("="*60 + "\n")
        f.write("Example DPO Samples\n")
        f.write("="*60 + "\n\n")
        for i in range(min(5, len(dpo_samples))):
            f.write(f"--- Sample {i+1} (Reward: {dpo_rewards[i]:.4f}, Density: {dpo_densities[i]:.4f}) ---\n")
            f.write(dpo_samples[i][:300] + "\n\n")
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Create comparison plots
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Reward comparison
        axes[0].hist(base_rewards, alpha=0.5, label='Base', bins=20)
        axes[0].hist(dpo_rewards, alpha=0.5, label='DPO', bins=20)
        axes[0].set_xlabel('Reward')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Reward Distribution')
        axes[0].legend()
        
        # Density comparison
        axes[1].hist(base_densities, alpha=0.5, label='Base', bins=20)
        axes[1].hist(dpo_densities, alpha=0.5, label='DPO', bins=20)
        axes[1].set_xlabel('Dialogue Density')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Dialogue Density Distribution')
        axes[1].legend()
        
        plt.tight_layout()
        plot_file = os.path.join(args.output_dir, 'dpo_comparison.png')
        plt.savefig(plot_file, dpi=150)
        print(f"✓ Plot saved to {plot_file}")
    except Exception as e:
        print(f"Could not create plots: {e}")


if __name__ == '__main__':
    main()

