"""
Prepare Linux Kernel dataset for nanoGPT training (PARALLELIZED VERSION).

This script:
1. Downloads the Linux kernel source code (if not already present)
2. Concatenates all .c and .h files into a single corpus (PARALLEL FILE READING)
3. Creates 5 subsets of different sizes: 100K, 500K, 1M, 5M, 10M characters (PARALLEL)
4. For each subset, creates train.bin (90%) and val.bin (10%)
5. Saves character vocabulary to meta.pkl

Optimizations:
- Uses ThreadPoolExecutor for parallel file I/O (reading ~70k files)
- Uses ProcessPoolExecutor for parallel subset creation (5 subsets)
- Estimated speedup: 3-5x faster than sequential version
- Expected time: 5-10 minutes (was 20-30 minutes)

Usage:
    python prepare.py
"""

import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import multiprocessing

# Dataset sizes to create (in characters)
SUBSET_SIZES = {
    '100k': 100_000,
    '500k': 500_000,
    '1m': 1_000_000,
    '5m': 5_000_000,
    '10m': 10_000_000,
}

def clone_linux_kernel(target_dir='linux_kernel_repo'):
    """Clone Linux kernel repository if not already present."""
    if os.path.exists(target_dir):
        print(f"Linux kernel repository already exists at {target_dir}")
        return target_dir
    
    print("Cloning Linux kernel repository (this will take a while)...")
    print("Repository size: ~3-4 GB")
    
    # Use shallow clone with depth=1 to save space and time
    cmd = [
        'git', 'clone',
        '--depth', '1',
        '--single-branch',
        '--branch', 'master',
        'git://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git',
        target_dir
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully cloned Linux kernel to {target_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        print("You can manually clone with:")
        print("  git clone --depth 1 git://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git linux_kernel_repo")
        sys.exit(1)
    
    return target_dir

def read_single_file(filepath):
    """Read a single source file. Helper for parallel processing."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read() + '\n\n'  # Separate files
    except Exception as e:
        return f"# Error reading {filepath}: {e}\n\n"


def collect_source_files(repo_dir, num_workers=None):
    """Find and concatenate all .c and .h files (parallelized)."""
    print(f"\nCollecting all .c and .h files from {repo_dir}...")
    
    source_files = []
    for root, dirs, files in os.walk(repo_dir):
        # Skip .git directory
        if '.git' in root:
            continue
        
        for file in files:
            if file.endswith('.c') or file.endswith('.h'):
                source_files.append(os.path.join(root, file))
    
    print(f"Found {len(source_files)} source files")
    
    # Parallelize file reading
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 16)  # Cap at 16 to avoid overhead
    
    print(f"Reading files in parallel (using {num_workers} workers)...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # ThreadPoolExecutor is better for I/O bound tasks
        contents = list(tqdm(
            executor.map(read_single_file, source_files),
            total=len(source_files),
            desc="Reading files"
        ))
    
    full_corpus = ''.join(contents)
    print(f"Total corpus size: {len(full_corpus):,} characters")
    
    return full_corpus

def create_subset(args):
    """
    Create a subset of specified size and split into train/val.
    Takes tuple (corpus, name, size, base_output_dir) for parallel processing.
    """
    corpus, name, size, base_output_dir = args
    output_dir = os.path.join(base_output_dir, name)
    
    print(f"\n[{name}] Creating {size:,} character subset...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract subset
    if size > len(corpus):
        print(f"[{name}] Warning: Requested size {size:,} exceeds corpus size {len(corpus):,}")
        print(f"[{name}] Using full corpus instead")
        subset = corpus
    else:
        subset = corpus[:size]
    
    print(f"[{name}] Subset size: {len(subset):,} characters")
    
    # Get all unique characters
    chars = sorted(list(set(subset)))
    vocab_size = len(chars)
    print(f"[{name}] Vocabulary size: {vocab_size} unique characters")
    
    # Create character mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Encode the subset
    data = np.array([stoi[ch] for ch in subset], dtype=np.uint16)
    
    # Split into train (90%) and val (10%)
    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]
    
    print(f"[{name}] Train: {len(train_data):,} chars, Val: {len(val_data):,} chars")
    
    # Save to binary files
    train_path = os.path.join(output_dir, 'train.bin')
    val_path = os.path.join(output_dir, 'val.bin')
    
    train_data.tofile(train_path)
    val_data.tofile(val_path)
    
    # Save metadata
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
        'data_size': len(subset),
    }
    
    meta_path = os.path.join(output_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"[{name}] ✓ Complete: train.bin, val.bin, meta.pkl")
    
    return (name, vocab_size)

def check_existing_subsets(output_dir):
    """Check which subsets already exist to avoid redundant work."""
    existing = {}
    for name in SUBSET_SIZES.keys():
        subset_dir = os.path.join(output_dir, name)
        train_path = os.path.join(subset_dir, 'train.bin')
        val_path = os.path.join(subset_dir, 'val.bin')
        meta_path = os.path.join(subset_dir, 'meta.pkl')
        
        if all(os.path.exists(p) for p in [train_path, val_path, meta_path]):
            existing[name] = True
        else:
            existing[name] = False
    
    return existing


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Prepare Linux Kernel dataset for nanoGPT')
    parser.add_argument('--output_dir', type=str, default='/nobackup/gaurav/kernel_code',
                       help='Directory to store prepared datasets (default: /nobackup/gaurav/kernel_code)')
    parser.add_argument('--repo_dir', type=str, default='linux_kernel_repo',
                       help='Directory for Linux kernel repository (default: linux_kernel_repo)')
    parser.add_argument('--cache_file', type=str, default='linux_kernel_corpus_cache.txt',
                       help='Path to corpus cache file (default: linux_kernel_corpus_cache.txt)')
    args = parser.parse_args()
    
    output_dir = args.output_dir
    repo_dir = args.repo_dir
    corpus_cache_path = args.cache_file
    
    print("="*60)
    print("Linux Kernel Dataset Preparation")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Repository directory: {repo_dir}")
    print(f"Corpus cache: {corpus_cache_path}")
    print("="*60)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check what already exists
    existing_subsets = check_existing_subsets(output_dir)
    all_exist = all(existing_subsets.values())
    
    if all_exist:
        print("\n✓ All subsets already exist!")
        for name in SUBSET_SIZES.keys():
            print(f"  - {output_dir}/{name}/ (train.bin, val.bin, meta.pkl)")
        
        response = input("\nRe-create all subsets? [y/N]: ").strip().lower()
        if response != 'y':
            print("\nSkipping data preparation. Existing subsets will be used.")
            print("To force re-creation, delete the subset directories or answer 'y'.")
            return
        else:
            print("\nRe-creating all subsets...")
    else:
        missing = [name for name, exists in existing_subsets.items() if not exists]
        existing = [name for name, exists in existing_subsets.items() if exists]
        
        if existing:
            print(f"\n✓ Found {len(existing)} existing subset(s):")
            for name in existing:
                print(f"  - {name}/")
        
        print(f"\n⚠ Need to create {len(missing)} subset(s):")
        for name in missing:
            print(f"  - {name}/")
        
        print("\nNote: Will skip existing subsets and only create missing ones.")
    
    # Step 1: Clone or locate Linux kernel repository
    repo_dir = clone_linux_kernel(repo_dir)
    
    # Check if we have a cached corpus to avoid re-reading
    
    if os.path.exists(corpus_cache_path):
        print(f"\n✓ Found cached corpus at {corpus_cache_path}")
        response = input("Use cached corpus (faster) or re-read from repo? [Y/n]: ").strip().lower()
        
        if response != 'n':
            print("Loading cached corpus...")
            with open(corpus_cache_path, 'r', encoding='utf-8') as f:
                corpus = f.read()
            print(f"Loaded corpus: {len(corpus):,} characters")
        else:
            corpus = collect_source_files(repo_dir)
            # Save corpus cache for next time
            print(f"\nSaving corpus cache to {corpus_cache_path}...")
            with open(corpus_cache_path, 'w', encoding='utf-8') as f:
                f.write(corpus)
    else:
        # Step 2: Collect and concatenate all source files
        corpus = collect_source_files(repo_dir)
        
        # Save corpus cache for next time
        print(f"\nSaving corpus cache to {corpus_cache_path}...")
        with open(corpus_cache_path, 'w', encoding='utf-8') as f:
            f.write(corpus)
        print("✓ Corpus cached for future runs")
    
    if len(corpus) < max(SUBSET_SIZES.values()):
        print(f"\nWarning: Corpus size ({len(corpus):,}) is smaller than largest subset ({max(SUBSET_SIZES.values()):,})")
        print("Proceeding anyway - largest subsets will use full corpus")
    
    # Step 3: Create subsets (in parallel, skip existing)
    print("\n" + "="*60)
    print("Creating dataset subsets")
    print("="*60)
    
    # Only process subsets that don't exist or need re-creation
    if all_exist and response == 'y':
        # User wants to re-create all
        subset_args = [(corpus, name, size, output_dir) for name, size in SUBSET_SIZES.items()]
    else:
        # Only create missing subsets
        existing_subsets = check_existing_subsets(output_dir)
        subset_args = [(corpus, name, size, output_dir) for name, size in SUBSET_SIZES.items() 
                       if not existing_subsets[name]]
        
        if not subset_args:
            print("✓ All subsets already exist, nothing to create!")
            vocab_sizes = {}
            for name in SUBSET_SIZES.keys():
                meta_path = os.path.join(output_dir, name, 'meta.pkl')
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                vocab_sizes[name] = meta['vocab_size']
        else:
            print(f"Processing {len(subset_args)} subset(s) in parallel...")
            
            # Use ProcessPoolExecutor for CPU-bound tasks (encoding, etc.)
            num_workers = min(len(subset_args), multiprocessing.cpu_count())
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(create_subset, subset_args))
            
            vocab_sizes = {name: size for name, size in results}
            
            # Add vocab sizes from existing subsets
            for name in SUBSET_SIZES.keys():
                if name not in vocab_sizes:
                    meta_path = os.path.join(output_dir, name, 'meta.pkl')
                    with open(meta_path, 'rb') as f:
                        meta = pickle.load(f)
                    vocab_sizes[name] = meta['vocab_size']
    
    # Summary
    print("\n" + "="*60)
    print("Dataset Preparation Complete!")
    print("="*60)
    print("\nCreated subsets:")
    for name, size in SUBSET_SIZES.items():
        print(f"  {name:>5}: {size:>10,} chars, vocab={vocab_sizes[name]:>3} chars, at {output_dir}/{name}/")
    
    print("\nNext steps:")
    print("  1. Review the data in each subset directory")
    print(f"  2. Run training: python run_data_scaling_experiment.py --num_gpus 8 --data_dir {output_dir}")
    print("  3. Analyze results: python analyze_data_scaling.py")

if __name__ == '__main__':
    main()

