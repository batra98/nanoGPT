"""
Generate samples from a trained Mamba model and compute evaluation metrics.

This is a variant of sample_and_evaluate.py that works with GPTMamba models.
"""

import os
import pickle
import torch
import numpy as np
from contextlib import nullcontext
from typing import List, Dict, Tuple
import argparse

from model_mamba import GPTConfig, GPTMamba  # ← Import Mamba model
from evaluation_metrics import EvaluationMetrics


def load_model(out_dir: str, device: str = 'cuda') -> Tuple[GPTMamba, callable, callable]:
    """
    Load trained Mamba model from checkpoint.
    
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
    model = GPTMamba(gptconf)  # ← Use GPTMamba
    
    # Load state dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    # Get encoder/decoder from meta.pkl
    config = checkpoint.get('config', {})
    dataset = config.get('dataset', 'shakespeare_char')
    
    # Determine data directory
    if os.path.isabs(dataset):
        data_dir = dataset
    else:
        data_dir = os.path.join('data', dataset)
    
    meta_path = os.path.join(data_dir, 'meta.pkl')
    
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi = meta['stoi']
        itos = meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # Fallback for datasets without meta.pkl
        print(f"Warning: meta.pkl not found in {data_dir}, using GPT-2 encoder")
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    
    return model, encode, decode


def generate_samples(
    model: GPTMamba,
    encode: callable,
    decode: callable,
    num_samples: int = 10,
    max_new_tokens: int = 1000,
    temperature: float = 0.8,
    top_k: int = 200,
    start_text: str = "\n",
    device: str = 'cuda'
) -> List[str]:
    """
    Generate text samples from the model.
    
    Args:
        model: Trained GPTMamba model
        encode: Function to encode text to tokens
        decode: Function to decode tokens to text
        num_samples: Number of samples to generate
        max_new_tokens: Maximum tokens per sample
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        start_text: Text to start generation from
        device: Device to use
    
    Returns:
        List of generated text samples
    """
    samples = []
    
    print(f"Generating {num_samples} samples...")
    
    with torch.no_grad():
        for i in range(num_samples):
            # Encode start text
            start_ids = encode(start_text)
            x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
            
            # Generate
            print(f"  Sample {i + 1}/{num_samples} - generating {max_new_tokens} tokens (this may take a while)...")
            import time
            start_time = time.time()
            
            with torch.no_grad():
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                generated_text = decode(y[0].tolist())
                samples.append(generated_text)
            
            elapsed = time.time() - start_time
            print(f"    ✓ Completed in {elapsed:.1f}s")
    
    print(f"✓ Generated {len(samples)} samples")
    return samples


def evaluate_model(
    out_dir: str,
    num_samples: int = 10,
    max_new_tokens: int = 1000,
    device: str = 'cuda',
    save_samples: bool = True
) -> Dict[str, float]:
    """
    Complete evaluation pipeline for a trained Mamba model.
    
    Args:
        out_dir: Directory containing trained model checkpoint
        num_samples: Number of samples to generate
        max_new_tokens: Maximum tokens per sample
        device: Device to use for generation
        save_samples: Whether to save generated samples to file
    
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING MAMBA MODEL: {out_dir}")
    print(f"{'='*70}")
    
    # Load model
    print("\n1. Loading model...")
    model, encode, decode = load_model(out_dir, device)
    
    # Get dataset path for metrics
    print("\n2. Loading training data...")
    checkpoint = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location='cpu')
    config = checkpoint.get('config', {})
    dataset = config.get('dataset', 'shakespeare_char')
    
    # Determine data directory
    if os.path.isabs(dataset):
        data_dir = dataset
    else:
        data_dir = os.path.join('data', dataset)
    
    # Check if samples already exist
    samples_path = os.path.join(out_dir, 'generated_samples.txt')
    
    if os.path.exists(samples_path):
        print(f"\n3. Loading existing samples from {samples_path}...")
        with open(samples_path, 'r') as f:
            content = f.read()
        
        # Parse samples (they're separated by "="*60 headers)
        generated_samples = []
        current_sample = []
        in_sample = False
        
        for line in content.split('\n'):
            if line.startswith('='*60):
                if current_sample and in_sample:
                    # End of a sample
                    generated_samples.append('\n'.join(current_sample))
                    current_sample = []
                    in_sample = False
            elif line.startswith('Sample '):
                in_sample = True
            elif in_sample and line.strip():
                current_sample.append(line)
        
        # Add last sample if exists
        if current_sample:
            generated_samples.append('\n'.join(current_sample))
        
        print(f"✓ Loaded {len(generated_samples)} existing samples")
        
        # If not enough samples, generate more
        if len(generated_samples) < num_samples:
            print(f"\n   Need {num_samples - len(generated_samples)} more samples...")
            additional_samples = generate_samples(
                model=model,
                encode=encode,
                decode=decode,
                num_samples=num_samples - len(generated_samples),
                max_new_tokens=max_new_tokens,
                device=device
            )
            generated_samples.extend(additional_samples)
    else:
        # Generate samples from scratch
        print(f"\n3. Generating {num_samples} samples...")
        generated_samples = generate_samples(
            model=model,
            encode=encode,
            decode=decode,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            device=device
        )
    
    # Save samples
    if save_samples:
        samples_path = os.path.join(out_dir, 'generated_samples.txt')
        with open(samples_path, 'w') as f:
            for i, sample in enumerate(generated_samples):
                f.write(f"{'='*60}\n")
                f.write(f"Sample {i+1}\n")
                f.write(f"{'='*60}\n")
                f.write(sample)
                f.write(f"\n\n")
        print(f"✓ Saved samples to {samples_path}")
    
    # Compute metrics
    print("\n4. Computing evaluation metrics...")
    evaluator = EvaluationMetrics(data_dir=data_dir)
    metrics = evaluator.compute_all_metrics(generated_samples)
    
    # Print metrics
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print("\nSpecific Metrics (training-data-dependent):")
    print(f"  1-gram overlap:  {metrics['1-gram_overlap']:.4f}")
    print(f"  2-gram overlap:  {metrics['2-gram_overlap']:.4f}")
    print(f"  3-gram overlap:  {metrics['3-gram_overlap']:.4f}")
    print(f"  Perplexity:      {metrics['perplexity']:.4f}")
    print(f"  KL divergence:   {metrics['kl_divergence']:.4f}")
    
    print("\nGeneral Metrics (training-data-independent):")
    print(f"  Self-BLEU:       {metrics['self_bleu']:.4f}")
    print(f"  Distinct-1:      {metrics['distinct_1']:.4f}")
    print(f"  Distinct-2:      {metrics['distinct_2']:.4f}")
    print(f"  Distinct-3:      {metrics['distinct_3']:.4f}")
    print(f"  Shannon entropy: {metrics['shannon_entropy']:.4f}")
    
    # Save metrics
    metrics_path = os.path.join(out_dir, 'evaluation_metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Saved metrics to {metrics_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained Mamba model')
    parser.add_argument('--out_dir', type=str, default='out-shakespeare-mamba',
                       help='Directory containing trained model')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to generate')
    parser.add_argument('--max_new_tokens', type=int, default=1000,
                       help='Maximum tokens per sample')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save samples to file')
    
    args = parser.parse_args()
    
    evaluate_model(
        out_dir=args.out_dir,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        save_samples=not args.no_save
    )


if __name__ == '__main__':
    main()

