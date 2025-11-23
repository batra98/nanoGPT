"""
Evaluate GRPO model and compare with base model.

Quantitative and qualitative comparison of base vs GRPO-aligned model.
"""

import os
import pickle
import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import GPTConfig, GPT
from verifier import compute_verifier_score, compute_verifier_scores, report_verifier_statistics


def get_encoder_decoder(checkpoint_path: str):
    """Get encode/decode functions from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
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
            return encode, decode
    
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    return encode, decode


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def generate_samples(model, prompts, encode, decode, num_samples=100, max_new_tokens=200, 
                     temperature=0.8, top_k=200, device='cuda', seed=42):
    """Generate samples from model."""
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed(seed)
    
    samples = []
    completion_texts = []
    
    print(f"Generating {num_samples} samples...")
    for i, prompt_array in enumerate(tqdm(prompts[:num_samples], desc="Generating")):
        prompt_text = decode(prompt_array.tolist())
        prompt_tensor = torch.tensor(
            prompt_array.astype(np.int64),
            dtype=torch.long,
            device=device
        )[None, ...]
        
        with torch.no_grad():
            completion = model.generate(
                prompt_tensor,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
        
        full_text = decode(completion[0].tolist())
        completion_text = full_text[len(prompt_text):]
        completion_texts.append(completion_text)
        samples.append(completion[0].cpu().numpy())
    
    return samples, completion_texts


def plot_comparison(base_scores, grpo_scores, base_reward_history=None, output_path='results/grpo_comparison.png'):
    """Plot comparison of base vs GRPO model."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2 if base_reward_history else 1, figsize=(12, 5))
    if base_reward_history:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Score distribution
    ax = axes[0]
    ax.hist(base_scores, bins=20, alpha=0.5, label='Base Model', color='blue')
    ax.hist(grpo_scores, bins=20, alpha=0.5, label='GRPO Model', color='green')
    ax.set_xlabel('Verifier Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Verifier Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training curve
    if base_reward_history:
        ax = axes[1]
        ax.plot(base_reward_history, label='Mean Reward', color='green')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Mean Verifier Reward')
        ax.set_title('GRPO Training Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✓ Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate GRPO model')
    parser.add_argument('--base_checkpoint', type=str, default='out-shakespeare/ckpt.pt',
                       help='Base model checkpoint')
    parser.add_argument('--grpo_checkpoint', type=str, default='out-grpo/ckpt.pt',
                       help='GRPO model checkpoint')
    parser.add_argument('--prompts_path', type=str, default='data/grpo/prompts.pkl',
                       help='Path to prompts')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to generate')
    parser.add_argument('--max_new_tokens', type=int, default=200,
                       help='Max new tokens')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=200,
                       help='Top-k sampling')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--wandb_log', action='store_true', default=False,
                       help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='grpo-rlvr',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print("GRPO Evaluation")
    print(f"{'='*60}\n")
    
    # Load prompts
    with open(args.prompts_path, 'rb') as f:
        prompts = pickle.load(f)
    print(f"Loaded {len(prompts)} prompts\n")
    
    # Load base model
    print("Loading base model...")
    base_model = load_model(args.base_checkpoint, args.device)
    base_encode, base_decode = get_encoder_decoder(args.base_checkpoint)
    print(f"✓ Base model loaded\n")
    
    # Load GRPO model
    print("Loading GRPO model...")
    grpo_model = load_model(args.grpo_checkpoint, args.device)
    grpo_encode, grpo_decode = get_encoder_decoder(args.grpo_checkpoint)
    print(f"✓ GRPO model loaded\n")
    
    # Generate samples from base model
    print("Generating samples from base model...")
    base_samples, base_texts = generate_samples(
        base_model, prompts, base_encode, base_decode,
        args.num_samples, args.max_new_tokens, args.temperature,
        args.top_k, args.device, seed=42
    )
    base_scores = compute_verifier_scores(base_texts)
    base_stats = report_verifier_statistics(base_scores)
    print(f"✓ Base model: Mean score = {base_stats['mean']:.2f}\n")
    
    # Generate samples from GRPO model
    print("Generating samples from GRPO model...")
    grpo_samples, grpo_texts = generate_samples(
        grpo_model, prompts, grpo_encode, grpo_decode,
        args.num_samples, args.max_new_tokens, args.temperature,
        args.top_k, args.device, seed=42
    )
    grpo_scores = compute_verifier_scores(grpo_texts)
    grpo_stats = report_verifier_statistics(grpo_scores)
    print(f"✓ GRPO model: Mean score = {grpo_stats['mean']:.2f}\n")
    
    # Comparison
    improvement = grpo_stats['mean'] - base_stats['mean']
    improvement_pct = (improvement / base_stats['mean']) * 100 if base_stats['mean'] > 0 else 0
    
    print(f"{'='*60}")
    print("Quantitative Comparison")
    print(f"{'='*60}")
    print(f"Base Model:")
    print(f"  Mean score: {base_stats['mean']:.2f}")
    print(f"  Std: {base_stats['std']:.2f}")
    print(f"  Min: {base_stats['min']:.2f}, Max: {base_stats['max']:.2f}")
    print(f"\nGRPO Model:")
    print(f"  Mean score: {grpo_stats['mean']:.2f}")
    print(f"  Std: {grpo_stats['std']:.2f}")
    print(f"  Min: {grpo_stats['min']:.2f}, Max: {grpo_stats['max']:.2f}")
    print(f"\nImprovement:")
    print(f"  Absolute: {improvement:+.2f}")
    print(f"  Relative: {improvement_pct:+.1f}%")
    print(f"{'='*60}\n")
    
    # Qualitative examples
    print(f"{'='*60}")
    print("Qualitative Examples")
    print(f"{'='*60}\n")
    
    # High-scoring examples
    base_high_idx = np.argmax(base_scores)
    grpo_high_idx = np.argmax(grpo_scores)
    
    print("--- High-Scoring Base Model Sample ---")
    print(f"Score: {base_scores[base_high_idx]:.2f}")
    prompt_text = base_decode(prompts[base_high_idx].tolist())
    print(f"Prompt: {prompt_text[:100]}...")
    print(f"Completion:\n{base_texts[base_high_idx][:300]}...\n")
    
    print("--- High-Scoring GRPO Model Sample ---")
    print(f"Score: {grpo_scores[grpo_high_idx]:.2f}")
    print(f"Prompt: {prompt_text[:100]}...")
    print(f"Completion:\n{grpo_texts[grpo_high_idx][:300]}...\n")
    
    # Low-scoring examples
    base_low_idx = np.argmin(base_scores)
    grpo_low_idx = np.argmin(grpo_scores)
    
    print("--- Low-Scoring Base Model Sample ---")
    print(f"Score: {base_scores[base_low_idx]:.2f}")
    prompt_text = base_decode(prompts[base_low_idx].tolist())
    print(f"Prompt: {prompt_text[:100]}...")
    print(f"Completion:\n{base_texts[base_low_idx][:300]}...\n")
    
    print("--- Low-Scoring GRPO Model Sample ---")
    print(f"Score: {grpo_scores[grpo_low_idx]:.2f}")
    print(f"Prompt: {prompt_text[:100]}...")
    print(f"Completion:\n{grpo_texts[grpo_low_idx][:300]}...\n")
    
    # Load reward history if available
    reward_history = None
    grpo_dir = os.path.dirname(args.grpo_checkpoint)
    reward_history_path = os.path.join(grpo_dir, 'reward_history.pkl')
    if os.path.exists(reward_history_path):
        with open(reward_history_path, 'rb') as f:
            reward_history = pickle.load(f)
    
    # Plot comparison
    plot_path = os.path.join(args.output_dir, 'grpo_comparison.png')
    plot_comparison(base_scores, grpo_scores, reward_history, plot_path)
    
    # Save results
    results = {
        'base_scores': base_scores,
        'grpo_scores': grpo_scores,
        'base_stats': base_stats,
        'grpo_stats': grpo_stats,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'base_texts': base_texts[:10],  # Save first 10 for examples
        'grpo_texts': grpo_texts[:10],
    }
    
    results_path = os.path.join(args.output_dir, 'grpo_evaluation.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Results saved to {results_path}\n")
    
    # Log to wandb
    if args.wandb_log:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"grpo-eval-{int(time.time())}",
            config={
                'base_checkpoint': args.base_checkpoint,
                'grpo_checkpoint': args.grpo_checkpoint,
                'num_samples': args.num_samples,
            }
        )
        
        wandb.log({
            'base/mean_score': base_stats['mean'],
            'base/std_score': base_stats['std'],
            'grpo/mean_score': grpo_stats['mean'],
            'grpo/std_score': grpo_stats['std'],
            'improvement/absolute': improvement,
            'improvement/percentage': improvement_pct,
        })
        
        # Log histograms
        wandb.log({
            'base/score_distribution': wandb.Histogram(base_scores),
            'grpo/score_distribution': wandb.Histogram(grpo_scores),
        })
        
        # Log plot
        wandb.log({'comparison_plot': wandb.Image(plot_path)})
        
        wandb.finish()
    
    print(f"{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

