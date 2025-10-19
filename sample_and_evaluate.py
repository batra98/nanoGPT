"""
Generate samples from a trained model and compute evaluation metrics.
"""

import os
import pickle
import torch
import numpy as np
from contextlib import nullcontext
from typing import List, Dict, Tuple
import argparse

from model import GPTConfig, GPT
from evaluation_metrics import EvaluationMetrics


def load_model(out_dir: str, device: str = 'cuda') -> Tuple[GPT, callable, callable]:
    """
    Load trained model from checkpoint.
    
    Args:
        out_dir: Directory containing ckpt.pt
        device: Device to load model on
    
    Returns:
        Tuple of (model, encode_fn, decode_fn)
    """
    # Load checkpoint
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Create model
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    # Load state dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    # Get encoder/decoder
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:
        dataset = checkpoint['config']['dataset']
        meta_path = os.path.join('data', dataset, 'meta.pkl')
        
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])
        else:
            raise FileNotFoundError(f"meta.pkl not found at {meta_path}")
    else:
        raise ValueError("Checkpoint does not contain dataset information")
    
    return model, encode, decode


def generate_samples(
    model: GPT,
    encode: callable,
    decode: callable,
    num_samples: int = 50,
    max_new_tokens: int = 500,
    temperature: float = 0.8,
    top_k: int = 200,
    device: str = 'cuda',
    seed: int = 1337,
    start_text: str = "\n"
) -> List[str]:
    """
    Generate multiple samples from the model.
    
    Args:
        model: Trained GPT model
        encode: Encoding function
        decode: Decoding function
        num_samples: Number of samples to generate
        max_new_tokens: Tokens per sample
        temperature: Sampling temperature
        top_k: Top-k sampling
        device: Device
        seed: Random seed
        start_text: Starting prompt
    
    Returns:
        List of generated text samples
    """
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed(seed)
    
    # Setup context for mixed precision
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Encode starting text
    start_ids = encode(start_text)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    # Generate samples
    samples = []
    with torch.no_grad():
        with ctx:
            for i in range(num_samples):
                # Reset to starting prompt for each sample
                x_sample = x.clone()
                y = model.generate(x_sample, max_new_tokens, temperature=temperature, top_k=top_k)
                sample_text = decode(y[0].tolist())
                samples.append(sample_text)
                
                if (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{num_samples} samples")
    
    return samples


def evaluate_model(
    out_dir: str,
    num_samples: int = 50,
    max_new_tokens: int = 500,
    device: str = 'cuda',
    save_samples: bool = True
) -> Dict[str, float]:
    """
    Load model, generate samples, and compute all metrics.
    
    Args:
        out_dir: Directory containing model checkpoint
        num_samples: Number of samples to generate
        max_new_tokens: Tokens per sample
        device: Device to use
        save_samples: Whether to save generated samples to file
    
    Returns:
        Dictionary of all metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating model from: {out_dir}")
    print(f"{'='*60}\n")
    
    # Load model
    print("Loading model...")
    model, encode, decode = load_model(out_dir, device)
    
    # Get validation loss from checkpoint
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    val_loss = checkpoint.get('best_val_loss', float('inf'))
    print(f"Validation loss: {val_loss:.4f}")
    
    # Generate samples
    print(f"\nGenerating {num_samples} samples...")
    samples = generate_samples(
        model, encode, decode,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        device=device
    )
    
    # Save samples if requested
    if save_samples:
        samples_path = os.path.join(out_dir, 'generated_samples.txt')
        with open(samples_path, 'w', encoding='utf-8') as f:
            for i, sample in enumerate(samples):
                f.write(f"{'='*60}\n")
                f.write(f"Sample {i+1}\n")
                f.write(f"{'='*60}\n")
                f.write(sample)
                f.write('\n\n')
        print(f"Saved samples to: {samples_path}")
    
    # Compute metrics
    print("\nComputing evaluation metrics...")
    
    # Determine dataset directory
    dataset = checkpoint['config'].get('dataset', 'shakespeare_char')
    data_dir = os.path.join('data', dataset)
    
    evaluator = EvaluationMetrics(data_dir=data_dir)
    metrics = evaluator.compute_all_metrics(samples, val_loss)
    
    # Print metrics
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print("\nSpecific Metrics (compare to training data):")
    print(f"  N-gram Overlap (1): {metrics['ngram_overlap_1']:.2f}%")
    print(f"  N-gram Overlap (2): {metrics['ngram_overlap_2']:.2f}%")
    print(f"  N-gram Overlap (3): {metrics['ngram_overlap_3']:.2f}%")
    print(f"  Perplexity:          {metrics['perplexity']:.4f}")
    print(f"  KL Divergence:       {metrics['kl_divergence']:.4f}")
    
    print("\nGeneral Metrics (no training data needed):")
    print(f"  Self-BLEU:           {metrics['self_bleu']:.2f}")
    print(f"  Distinct-1:          {metrics['distinct_1']:.2f}%")
    print(f"  Distinct-2:          {metrics['distinct_2']:.2f}%")
    print(f"  Distinct-3:          {metrics['distinct_3']:.2f}%")
    print(f"  Entropy:             {metrics['entropy']:.4f} bits")
    print("="*60 + "\n")
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate samples and evaluate model')
    parser.add_argument('--out_dir', type=str, default='out-shakespeare-char',
                        help='Directory containing model checkpoint')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples to generate')
    parser.add_argument('--max_new_tokens', type=int, default=500,
                        help='Number of tokens per sample')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save generated samples')
    
    args = parser.parse_args()
    
    metrics = evaluate_model(
        out_dir=args.out_dir,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        save_samples=not args.no_save
    )

