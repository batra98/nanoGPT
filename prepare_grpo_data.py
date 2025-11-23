"""
Prepare prompts and evaluate baseline for GRPO training.

1. Extract prompts from validation set
2. Generate baseline completions from base model
3. Compute verifier scores
4. Report statistics and examples
"""

import os
import pickle
import argparse
import numpy as np
import torch
from tqdm import tqdm

from model import GPTConfig, GPT
from verifier import compute_verifier_score, compute_verifier_scores, report_verifier_statistics


def get_encoder_decoder(checkpoint_path: str):
    """Get encode/decode functions from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to load meta.pkl for character-level encoding
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
    
    # Fall back to BPE
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    return encode, decode


def extract_prompts_from_validation(num_prompts: int = 100, prompt_length: int = 64):
    """
    Extract prompts from validation set.
    
    Args:
        num_prompts: Number of prompts to extract
        prompt_length: Length of each prompt in tokens
    
    Returns:
        List of prompt token arrays
    """
    # Load validation data
    val_data_path = 'data/shakespeare/val.bin'
    if not os.path.exists(val_data_path):
        raise FileNotFoundError(f"Validation data not found at {val_data_path}")
    
    val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')
    
    # Extract non-overlapping prompts
    prompts = []
    idx = 0
    
    while len(prompts) < num_prompts and idx + prompt_length < len(val_data):
        prompt = val_data[idx:idx + prompt_length].astype(np.int64)
        prompts.append(prompt)
        idx += prompt_length  # Non-overlapping
    
    if len(prompts) < num_prompts:
        print(f"Warning: Only extracted {len(prompts)} prompts (requested {num_prompts})")
    
    return prompts[:num_prompts]


def generate_completion(model, prompt_tensor, max_new_tokens=200, temperature=0.8, top_k=200, device='cuda'):
    """Generate completion from model."""
    model.eval()
    with torch.no_grad():
        completion = model.generate(
            prompt_tensor,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    return completion


def main():
    parser = argparse.ArgumentParser(description='Prepare GRPO data and evaluate baseline')
    parser.add_argument('--checkpoint_dir', type=str, default='out-shakespeare',
                       help='Directory containing base model checkpoint')
    parser.add_argument('--num_prompts', type=int, default=100,
                       help='Number of prompts to extract')
    parser.add_argument('--prompt_length', type=int, default=64,
                       help='Length of each prompt in tokens')
    parser.add_argument('--max_new_tokens', type=int, default=200,
                       help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=200,
                       help='Top-k sampling')
    parser.add_argument('--output_dir', type=str, default='data/grpo',
                       help='Output directory for prompts and results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print("GRPO Data Preparation and Baseline Evaluation")
    print(f"{'='*60}\n")
    
    # Step 1: Extract prompts
    print(f"Step 1: Extracting {args.num_prompts} prompts from validation set...")
    prompts = extract_prompts_from_validation(args.num_prompts, args.prompt_length)
    print(f"✓ Extracted {len(prompts)} prompts\n")
    
    # Save prompts
    prompts_path = os.path.join(args.output_dir, 'prompts.pkl')
    with open(prompts_path, 'wb') as f:
        pickle.dump(prompts, f)
    print(f"✓ Saved prompts to {prompts_path}\n")
    
    # Step 2: Load base model
    checkpoint_path = os.path.join(args.checkpoint_dir, 'ckpt.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    print(f"Step 2: Loading base model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()
    
    encode, decode = get_encoder_decoder(checkpoint_path)
    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters\n")
    
    # Step 3: Generate baseline completions
    print(f"Step 3: Generating {len(prompts)} baseline completions...")
    completions = []
    completion_texts = []
    
    for i, prompt_array in enumerate(tqdm(prompts, desc="Generating")):
        prompt_tensor = torch.tensor(
            prompt_array.astype(np.int64),
            dtype=torch.long,
            device=args.device
        )[None, ...]
        
        completion = generate_completion(
            model, prompt_tensor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device
        )
        
        completions.append(completion[0].cpu().numpy())
        
        # Decode to text
        prompt_text = decode(prompt_array.tolist())
        full_text = decode(completion[0].tolist())
        completion_text = full_text[len(prompt_text):]  # Only generated part
        completion_texts.append(completion_text)
    
    print(f"✓ Generated {len(completions)} completions\n")
    
    # Step 4: Compute verifier scores
    print("Step 4: Computing verifier scores...")
    scores = compute_verifier_scores(completion_texts)
    stats = report_verifier_statistics(scores)
    
    print(f"\n{'='*60}")
    print("Baseline Verifier Statistics")
    print(f"{'='*60}")
    print(f"Mean score: {stats['mean']:.2f}")
    print(f"Std deviation: {stats['std']:.2f}")
    print(f"Min score: {stats['min']:.2f}")
    print(f"Max score: {stats['max']:.2f}")
    print(f"Median score: {stats['median']:.2f}")
    print(f"Total samples: {stats['count']}")
    print(f"{'='*60}\n")
    
    # Step 5: Find representative examples
    print("Step 5: Finding representative examples...")
    scores_array = np.array(scores)
    
    # High, medium, low examples
    high_idx = np.argmax(scores_array)
    low_idx = np.argmin(scores_array)
    median_idx = np.argsort(scores_array)[len(scores_array) // 2]
    
    # Find examples near quartiles
    sorted_indices = np.argsort(scores_array)
    q1_idx = sorted_indices[len(sorted_indices) // 4]
    q3_idx = sorted_indices[3 * len(sorted_indices) // 4]
    
    examples = [
        ('Highest', high_idx, scores[high_idx]),
        ('Q3 (75th percentile)', q3_idx, scores[q3_idx]),
        ('Median', median_idx, scores[median_idx]),
        ('Q1 (25th percentile)', q1_idx, scores[q1_idx]),
        ('Lowest', low_idx, scores[low_idx]),
    ]
    
    print(f"\n{'='*60}")
    print("Representative Examples")
    print(f"{'='*60}\n")
    
    for label, idx, score in examples:
        prompt_text = decode(prompts[idx].tolist())
        print(f"--- {label} (score={score:.2f}) ---")
        print(f"Prompt: {prompt_text[:100]}...")
        print(f"Completion:\n{completion_texts[idx][:300]}...")
        print()
    
    # Step 6: Save baseline results
    baseline_results = {
        'prompts': prompts,
        'completions': completions,
        'completion_texts': completion_texts,
        'scores': scores,
        'stats': stats,
        'examples': examples
    }
    
    baseline_path = os.path.join(args.output_dir, 'baseline_results.pkl')
    with open(baseline_path, 'wb') as f:
        pickle.dump(baseline_results, f)
    print(f"✓ Saved baseline results to {baseline_path}\n")
    
    print(f"{'='*60}")
    print("Baseline Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Prompts saved to: {prompts_path}")
    print(f"Baseline results saved to: {baseline_path}")

if __name__ == '__main__':
    main()

