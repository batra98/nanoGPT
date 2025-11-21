"""
Generate preference pairs for reward model training.

This script:
1. Loads the Linux kernel pretrained checkpoint
2. Samples pairs of completions for the same prompts
3. Uses preference heuristic to label which is better
4. Saves training data for reward model
"""

import os
import pickle
import argparse
from contextlib import nullcontext
from typing import List, Tuple, Dict

import torch
import numpy as np
from tqdm import tqdm

from model import GPTConfig, GPT
from preference_heuristic import assign_preference, compute_comment_density, report_statistics


def load_model_and_data(checkpoint_dir: str, device: str = 'cuda'):
    """Load trained model from checkpoint."""
    ckpt_path = os.path.join(checkpoint_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    print(f"Loading checkpoint from {ckpt_path}...")
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
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Load encoder/decoder
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:
        dataset = checkpoint['config']['dataset']
        # Handle both absolute and relative dataset paths
        if os.path.isabs(dataset):
            meta_path = os.path.join(dataset, 'meta.pkl')
        else:
            meta_path = os.path.join('data', dataset, 'meta.pkl')
        
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])
            print(f"Loaded character-level vocabulary from {meta_path} (size={len(stoi)})")
        else:
            # Fall back to tiktoken for BPE datasets
            print(f"No meta.pkl found at {meta_path}, using tiktoken GPT-2 BPE encoding")
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            decode = lambda l: enc.decode(l)
            print(f"Loaded BPE vocabulary (size=50257)")
    else:
        # If no dataset info, assume GPT-2 BPE
        print("No dataset info in checkpoint, using tiktoken GPT-2 BPE encoding")
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
        print(f"Loaded BPE vocabulary (size=50257)")
    
    # Load validation data for prompts
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:
        dataset = checkpoint['config']['dataset']
        if os.path.isabs(dataset):
            val_data_path = os.path.join(dataset, 'val.bin')
        else:
            val_data_path = os.path.join('data', dataset, 'val.bin')
    else:
        # Default to shakespeare if no dataset info
        val_data_path = 'data/shakespeare/val.bin'
    
    if not os.path.exists(val_data_path):
        raise FileNotFoundError(f"Validation data not found at {val_data_path}")
    
    val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')
    print(f"Loaded validation data from {val_data_path} ({len(val_data):,} tokens)")
    
    return model, encode, decode, val_data


def get_prompts_from_data(val_data: np.ndarray, 
                          num_prompts: int, 
                          prompt_length: int = 128,
                          seed: int = 42) -> List[np.ndarray]:
    """Extract random prompts from validation data."""
    np.random.seed(seed)
    prompts = []
    
    max_start = len(val_data) - prompt_length - 1
    for _ in range(num_prompts):
        start_idx = np.random.randint(0, max_start)
        prompt = val_data[start_idx:start_idx + prompt_length]
        prompts.append(prompt)
    
    return prompts


def generate_completion(model: GPT, 
                       prompt: torch.Tensor,
                       max_new_tokens: int = 256,
                       temperature: float = 0.8,
                       top_k: int = 200,
                       device: str = 'cuda') -> torch.Tensor:
    """Generate a completion from the model."""
    with torch.no_grad():
        completion = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    return completion


def generate_preference_pairs(model: GPT,
                              prompts: List[np.ndarray],
                              decode: callable,
                              max_new_tokens: int = 256,
                              device: str = 'cuda') -> List[Dict]:
    """
    Generate preference pairs by sampling two completions per prompt.
    
    Uses different temperatures to get diverse completions.
    """
    preference_data = []
    
    print(f"\nGenerating preference pairs for {len(prompts)} prompts...")
    
    for prompt_array in tqdm(prompts):
        # Convert prompt to tensor
        prompt_tensor = torch.tensor(
            prompt_array.astype(np.int64), 
            dtype=torch.long, 
            device=device
        )[None, ...]
        
        # Generate two completions with different sampling strategies
        # Completion A: moderate temperature
        completion_a = generate_completion(
            model, prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=0.9,
            top_k=200,
            device=device
        )
        
        # Completion B: different temperature for diversity
        completion_b = generate_completion(
            model, prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_k=150,
            device=device
        )
        
        # Decode to text
        prompt_text = decode(prompt_array.tolist())
        text_a = decode(completion_a[0].tolist())
        text_b = decode(completion_b[0].tolist())
        
        # Get only the generated part (remove prompt)
        generated_a = text_a[len(prompt_text):]
        generated_b = text_b[len(prompt_text):]
        
        # Assign preference based on heuristic
        preference, density_a, density_b = assign_preference(generated_a, generated_b)
        
        preference_data.append({
            'prompt': prompt_text,
            'completion_a': generated_a,
            'completion_b': generated_b,
            'full_a': text_a,
            'full_b': text_b,
            'preference': preference,  # 0 if A preferred, 1 if B preferred
            'density_a': density_a,
            'density_b': density_b,
        })
    
    return preference_data


def main():
    parser = argparse.ArgumentParser(description='Generate preference pairs for reward model training')
    parser.add_argument('--checkpoint_dir', type=str, default='out-shakespeare',
                       help='Directory containing model checkpoint')
    parser.add_argument('--num_pairs', type=int, default=1000,
                       help='Number of preference pairs to generate')
    parser.add_argument('--prompt_length', type=int, default=64,
                       help='Length of prompt in tokens')
    parser.add_argument('--max_new_tokens', type=int, default=128,
                       help='Maximum new tokens to generate')
    parser.add_argument('--output_dir', type=str, default='data/preferences',
                       help='Directory to save preference data')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load model and data
    model, encode, decode, val_data = load_model_and_data(args.checkpoint_dir, args.device)
    
    # Extract prompts
    print(f"\nExtracting {args.num_pairs} prompts from validation data...")
    prompts = get_prompts_from_data(
        val_data,
        num_prompts=args.num_pairs,
        prompt_length=args.prompt_length,
        seed=args.seed
    )
    
    # Generate preference pairs
    preference_data = generate_preference_pairs(
        model,
        prompts,
        decode,
        max_new_tokens=args.max_new_tokens,
        device=args.device
    )
    
    # Report statistics
    print(f"\n{'='*60}")
    print("Preference Data Statistics")
    print(f"{'='*60}")
    print(f"Total pairs generated: {len(preference_data)}")
    
    num_a_preferred = sum(1 for d in preference_data if d['preference'] == 0)
    num_b_preferred = sum(1 for d in preference_data if d['preference'] == 1)
    print(f"A preferred: {num_a_preferred} ({num_a_preferred/len(preference_data)*100:.1f}%)")
    print(f"B preferred: {num_b_preferred} ({num_b_preferred/len(preference_data)*100:.1f}%)")
    
    # Report comment density statistics
    all_completions_a = [d['completion_a'] for d in preference_data]
    all_completions_b = [d['completion_b'] for d in preference_data]
    
    report_statistics(all_completions_a, "Completion A (temp=0.9)")
    report_statistics(all_completions_b, "Completion B (temp=0.7)")
    
    # Split into train/val (80/20)
    split_idx = int(0.8 * len(preference_data))
    train_data = preference_data[:split_idx]
    val_data_pref = preference_data[split_idx:]
    
    print(f"\nSplit: {len(train_data)} train, {len(val_data_pref)} validation")
    
    # Save to disk
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_path = os.path.join(args.output_dir, 'train.pkl')
    val_path = os.path.join(args.output_dir, 'val.pkl')
    
    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)
    print(f"Saved training data to {train_path}")
    
    with open(val_path, 'wb') as f:
        pickle.dump(val_data_pref, f)
    print(f"Saved validation data to {val_path}")
    
    # Show some examples
    print(f"\n{'='*60}")
    print("Example Preference Pairs")
    print(f"{'='*60}")
    
    for i in range(min(3, len(preference_data))):
        example = preference_data[i]
        print(f"\n--- Example {i+1} ---")
        print(f"Prompt: {example['prompt'][:100]}...")
        print(f"\nCompletion A (density={example['density_a']:.4f}):")
        print(example['completion_a'][:200])
        print(f"\nCompletion B (density={example['density_b']:.4f}):")
        print(example['completion_b'][:200])
        print(f"\nPreferred: {'A' if example['preference'] == 0 else 'B'}")
        print("-" * 60)


if __name__ == '__main__':
    main()

