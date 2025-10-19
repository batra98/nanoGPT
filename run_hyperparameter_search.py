"""
Orchestrate hyperparameter search for Shakespeare character-level model.
Tests different combinations of n_layer, n_head, and n_embd.
Logs all results to WandB with identifiable run names.
"""

import os
import sys
import time
import subprocess
import csv
from typing import Dict, List, Tuple
import torch

from sample_and_evaluate import evaluate_model


# Configuration space: (n_layer, n_head, n_embd, description)
CONFIGURATIONS = [
    (4, 4, 256, "smallest-fastest"),
    (6, 6, 384, "baseline-original"),
    (8, 8, 384, "deeper-more-heads"),
    (6, 4, 512, "wider-embeddings"),
    (8, 4, 256, "deep-narrow"),
    (6, 8, 512, "more-heads-wider"),
]


def count_parameters(n_layer: int, n_head: int, n_embd: int, 
                     vocab_size: int = 65, block_size: int = 256) -> int:
    """
    Estimate the number of parameters in the model.
    
    Args:
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        vocab_size: Vocabulary size
        block_size: Context length
    
    Returns:
        Approximate number of parameters
    """
    # Token embeddings + position embeddings
    embedding_params = vocab_size * n_embd + block_size * n_embd
    
    # Per transformer block:
    # - Attention: 3 * n_embd * n_embd (QKV) + n_embd * n_embd (output projection)
    # - MLP: n_embd * 4*n_embd (up) + 4*n_embd * n_embd (down)
    # - LayerNorm: 2 * n_embd (weight + bias) * 2 (two layer norms per block)
    attention_params = 4 * n_embd * n_embd
    mlp_params = 2 * n_embd * 4 * n_embd
    layernorm_params = 4 * n_embd
    block_params = (attention_params + mlp_params + layernorm_params) * n_layer
    
    # Final layer norm + output head
    output_params = n_embd + vocab_size * n_embd
    
    total_params = embedding_params + block_params + output_params
    return total_params


def train_configuration(
    n_layer: int,
    n_head: int,
    n_embd: int,
    description: str,
    base_config: str = "config/hyperparam_search_base.py",
    device: str = "cuda",
    num_gpus: int = 1
) -> Tuple[str, float, Dict]:
    """
    Train a single configuration.
    
    Args:
        n_layer: Number of layers
        n_head: Number of heads
        n_embd: Embedding dimension
        description: Description for run name
        base_config: Base configuration file
        device: Device to train on
        num_gpus: Number of GPUs to use (1 for single GPU, 8 for DDP)
    
    Returns:
        Tuple of (out_dir, training_time, metrics_dict)
    """
    # Create run name
    run_name = f"shakespeare-L{n_layer}-H{n_head}-E{n_embd}"
    out_dir = f"out-shakespeare-hyperparam-L{n_layer}-H{n_head}-E{n_embd}"
    
    # Count parameters
    total_params = count_parameters(n_layer, n_head, n_embd)
    
    print("\n" + "="*80)
    print(f"TRAINING CONFIGURATION: {run_name}")
    print("="*80)
    print(f"  Layers:     {n_layer}")
    print(f"  Heads:      {n_head}")
    print(f"  Embedding:  {n_embd}")
    print(f"  Parameters: {total_params:,} (~{total_params/1e6:.2f}M)")
    print(f"  GPUs:       {num_gpus}")
    print(f"  Output dir: {out_dir}")
    print("="*80 + "\n")
    
    # Build training command (use torchrun for multi-GPU, python for single GPU)
    if num_gpus > 1:
        # Multi-GPU training with DDP
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={num_gpus}",
            "train.py",
            base_config,
            f"--out_dir={out_dir}",
            f"--wandb_run_name={run_name}",
            f"--n_layer={n_layer}",
            f"--n_head={n_head}",
            f"--n_embd={n_embd}",
        ]
    else:
        # Single GPU training
        cmd = [
            sys.executable,
            "train.py",
            base_config,
            f"--out_dir={out_dir}",
            f"--wandb_run_name={run_name}",
            f"--n_layer={n_layer}",
            f"--n_head={n_head}",
            f"--n_embd={n_embd}",
            f"--device={device}",
        ]
    
    # Run training
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        training_time = (time.time() - start_time) / 60  # Convert to minutes
        print(f"\n✓ Training completed in {training_time:.2f} minutes")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error code {e.returncode}")
        raise
    
    return out_dir, training_time, {}


def main():
    """Run hyperparameter search."""
    import argparse
    parser = argparse.ArgumentParser(description='Run hyperparameter search')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to use per configuration (1 or 8)')
    args = parser.parse_args()
    
    num_gpus = args.num_gpus
    
    print("\n" + "="*80)
    print("SHAKESPEARE CHARACTER-LEVEL MODEL HYPERPARAMETER SEARCH")
    print("="*80)
    print(f"Total configurations to test: {len(CONFIGURATIONS)}")
    print(f"GPUs per configuration: {num_gpus}")
    expected_time_per_config = 8 if num_gpus == 1 else 2
    print(f"Expected time per config: ~{expected_time_per_config} minutes")
    print(f"Expected total time: ~{len(CONFIGURATIONS) * expected_time_per_config} minutes")
    print("="*80 + "\n")
    
    # Check if data is prepared
    data_dir = "data/shakespeare_char"
    if not os.path.exists(os.path.join(data_dir, "train.bin")):
        print("ERROR: Shakespeare data not found!")
        print(f"Please run: python {data_dir}/prepare.py")
        return
    
    # Check device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: Training on CPU will be very slow!")
    
    if num_gpus > 1:
        if not torch.cuda.is_available():
            print("ERROR: Multi-GPU training requested but CUDA not available!")
            return
        gpu_count = torch.cuda.device_count()
        print(f"Available GPUs: {gpu_count}")
        if gpu_count < num_gpus:
            print(f"WARNING: Requested {num_gpus} GPUs but only {gpu_count} available!")
            print(f"Using {gpu_count} GPUs instead.")
            num_gpus = gpu_count
    print()
    
    # Results storage
    all_results = []
    
    # Train each configuration
    for i, (n_layer, n_head, n_embd, description) in enumerate(CONFIGURATIONS, 1):
        print(f"\n{'#'*80}")
        print(f"# Configuration {i}/{len(CONFIGURATIONS)}: {description}")
        print(f"{'#'*80}\n")
        
        try:
            # Train model
            out_dir, training_time, _ = train_configuration(
                n_layer=n_layer,
                n_head=n_head,
                n_embd=n_embd,
                description=description,
                device=device,
                num_gpus=num_gpus
            )
            
            # Generate samples and evaluate
            print(f"\nEvaluating configuration {i}/{len(CONFIGURATIONS)}...")
            metrics = evaluate_model(
                out_dir=out_dir,
                num_samples=50,
                max_new_tokens=500,
                device=device,
                save_samples=True
            )
            
            # Load checkpoint to get losses
            ckpt_path = os.path.join(out_dir, 'ckpt.pt')
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                train_loss = checkpoint.get('config', {}).get('train_loss', 'N/A')
                val_loss = checkpoint.get('best_val_loss', 'N/A')
            else:
                train_loss = 'N/A'
                val_loss = 'N/A'
            
            # Store results
            result = {
                'config_name': f"L{n_layer}-H{n_head}-E{n_embd}",
                'description': description,
                'n_layer': n_layer,
                'n_head': n_head,
                'n_embd': n_embd,
                'total_params': count_parameters(n_layer, n_head, n_embd),
                'training_time_min': training_time,
                'train_loss': train_loss,
                'val_loss': val_loss,
                **metrics
            }
            all_results.append(result)
            
            print(f"\n✓ Configuration {i}/{len(CONFIGURATIONS)} complete!")
            
        except Exception as e:
            print(f"\n✗ Configuration {i}/{len(CONFIGURATIONS)} failed: {str(e)}")
            # Continue with next configuration
            continue
    
    # Save results summary
    if all_results:
        csv_path = "results_summary.csv"
        print(f"\n{'='*80}")
        print("SAVING RESULTS SUMMARY")
        print(f"{'='*80}\n")
        
        with open(csv_path, 'w', newline='') as f:
            if all_results:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
        
        print(f"Results saved to: {csv_path}")
        
        # Print summary table
        print(f"\n{'='*80}")
        print("RESULTS SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"{'Config':<15} {'Params':<10} {'Time(min)':<10} {'Val Loss':<10} {'Perplexity':<12}")
        print("-" * 80)
        
        for result in all_results:
            config = result['config_name']
            params = f"{result['total_params']/1e6:.1f}M"
            time_min = f"{result['training_time_min']:.1f}"
            val_loss = f"{result['val_loss']:.4f}" if isinstance(result['val_loss'], float) else str(result['val_loss'])
            perplexity = f"{result['perplexity']:.4f}" if 'perplexity' in result else 'N/A'
            
            print(f"{config:<15} {params:<10} {time_min:<10} {val_loss:<10} {perplexity:<12}")
        
        # Find best configuration
        valid_results = [r for r in all_results if isinstance(r['val_loss'], float)]
        if valid_results:
            best_result = min(valid_results, key=lambda x: x['val_loss'])
            
            print("\n" + "="*80)
            print("BEST CONFIGURATION")
            print("="*80)
            print(f"  Config:      {best_result['config_name']} ({best_result['description']})")
            print(f"  Layers:      {best_result['n_layer']}")
            print(f"  Heads:       {best_result['n_head']}")
            print(f"  Embedding:   {best_result['n_embd']}")
            print(f"  Parameters:  {best_result['total_params']:,} (~{best_result['total_params']/1e6:.2f}M)")
            print(f"  Val Loss:    {best_result['val_loss']:.4f}")
            print(f"  Perplexity:  {best_result['perplexity']:.4f}")
            print("="*80 + "\n")
    
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH COMPLETE!")
    print("="*80)
    print(f"Total configurations tested: {len(all_results)}")
    print(f"Results saved to: results_summary.csv")
    print(f"Check WandB project: shakespeare-hyperparam-search")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

