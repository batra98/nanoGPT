"""
Extensive hyperparameter search with early stopping.
Tests a larger grid of configurations with adaptive iteration counts.
"""

import os
import sys
import time
import subprocess
import csv
from typing import Dict, List, Tuple
import torch

from sample_and_evaluate import evaluate_model

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Metrics will not be logged to WandB.")


# EXTENSIVE configuration space: (n_layer, n_head, n_embd, description)
# Testing wider range of architectures
CONFIGURATIONS = [
    # Small models (fast to train)
    (4, 4, 256, "tiny-256"),
    (4, 4, 384, "tiny-384"),
    (4, 4, 512, "tiny-512"),
    
    # Medium-small models
    (6, 4, 256, "medium-narrow"),
    (6, 4, 384, "medium-384"),
    (6, 4, 512, "medium-512"),
    
    # Baseline and variations
    (6, 6, 256, "baseline-narrow"),
    (6, 6, 384, "baseline-original"),
    (6, 6, 512, "baseline-wide"),
    
    # More heads
    (6, 8, 256, "more-heads-narrow"),
    (6, 8, 384, "more-heads-384"),
    (6, 8, 512, "more-heads-wide"),
    
    # Deeper models
    (8, 4, 256, "deep-narrow"),
    (8, 4, 384, "deep-384"),
    (8, 4, 512, "deep-512"),
    
    (8, 8, 256, "deep-wide-narrow"),
    (8, 8, 384, "deep-wide-384"),
    (8, 8, 512, "deep-wide-512"),
    
    # Very deep
    (10, 6, 384, "very-deep-balanced"),
    (10, 8, 512, "very-deep-wide"),
]


def count_parameters(n_layer: int, n_head: int, n_embd: int, 
                     vocab_size: int = 65, block_size: int = 256) -> int:
    """Estimate the number of parameters in the model."""
    embedding_params = vocab_size * n_embd + block_size * n_embd
    attention_params = 4 * n_embd * n_embd
    mlp_params = 2 * n_embd * 4 * n_embd
    layernorm_params = 4 * n_embd
    block_params = (attention_params + mlp_params + layernorm_params) * n_layer
    output_params = n_embd + vocab_size * n_embd
    total_params = embedding_params + block_params + output_params
    return total_params


def train_configuration(
    n_layer: int,
    n_head: int,
    n_embd: int,
    description: str,
    base_config: str = "config/hyperparam_search_early_stopping.py",
    device: str = "cuda",
    num_gpus: int = 1
) -> Tuple[str, float, Dict]:
    """
    Train a single configuration with early stopping.
    """
    # Create run name
    run_name = f"shakespeare-L{n_layer}-H{n_head}-E{n_embd}"
    out_dir = f"out-shakespeare-extensive-L{n_layer}-H{n_head}-E{n_embd}"
    
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
    print(f"  Early stopping: ENABLED (patience=5, delta=0.001)")
    print("="*80 + "\n")
    
    # Build training command with early stopping script
    if num_gpus > 1:
        # Multi-GPU training with DDP
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={num_gpus}",
            "train_with_early_stopping.py",
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
            "train_with_early_stopping.py",
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
            capture_output=False,
            text=True
        )
        training_time = (time.time() - start_time) / 60
        print(f"\n✓ Training completed in {training_time:.2f} minutes")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error code {e.returncode}")
        raise
    
    return out_dir, training_time, {}


def main():
    """Run extensive hyperparameter search with early stopping."""
    import argparse
    parser = argparse.ArgumentParser(description='Run extensive hyperparameter search')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to use per configuration (1 or 8)')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples to generate for evaluation')
    args = parser.parse_args()
    
    num_gpus = args.num_gpus
    
    print("\n" + "="*80)
    print("EXTENSIVE SHAKESPEARE HYPERPARAMETER SEARCH WITH EARLY STOPPING")
    print("="*80)
    print(f"Total configurations to test: {len(CONFIGURATIONS)}")
    print(f"GPUs per configuration: {num_gpus}")
    print(f"Max iterations per config: 10,000 (with early stopping)")
    print(f"Early stopping: patience=5, min_delta=0.001")
    print(f"Expected time: VARIABLE (depends on convergence)")
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
            
            # Wait a moment and clear GPU memory after DDP training
            if num_gpus > 1:
                print("\nWaiting for DDP processes to fully terminate...")
                time.sleep(3)  # Give DDP processes time to clean up
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Generate samples and evaluate (always on single GPU for consistency)
            print(f"\nEvaluating configuration {i}/{len(CONFIGURATIONS)}...")
            eval_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            metrics = evaluate_model(
                out_dir=out_dir,
                num_samples=args.num_samples,
                max_new_tokens=500,
                device=eval_device,  # Use single GPU for evaluation
                save_samples=True
            )
            
            # Load checkpoint to get losses and iteration count
            ckpt_path = os.path.join(out_dir, 'ckpt.pt')
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                train_loss = checkpoint.get('config', {}).get('train_loss', 'N/A')
                val_loss = checkpoint.get('best_val_loss', 'N/A')
                final_iter = checkpoint.get('iter_num', 'N/A')
            else:
                train_loss = 'N/A'
                val_loss = 'N/A'
                final_iter = 'N/A'
            
            # Store results
            result = {
                'config_name': f"L{n_layer}-H{n_head}-E{n_embd}",
                'description': description,
                'n_layer': n_layer,
                'n_head': n_head,
                'n_embd': n_embd,
                'total_params': count_parameters(n_layer, n_head, n_embd),
                'final_iteration': final_iter,
                'training_time_min': training_time,
                'train_loss': train_loss,
                'val_loss': val_loss,
                **metrics
            }
            all_results.append(result)
            
            # Log evaluation metrics to WandB
            if WANDB_AVAILABLE:
                run_name = f"shakespeare-L{n_layer}-H{n_head}-E{n_embd}"
                try:
                    run = wandb.init(
                        project='shakespeare-extensive-search',
                        name=run_name,
                        id=run_name,
                        resume='allow'
                    )
                    
                    wandb.log({
                        'config/n_layer': n_layer,
                        'config/n_head': n_head,
                        'config/n_embd': n_embd,
                        'config/total_params': result['total_params'],
                        'config/training_time_min': training_time,
                        'config/final_iteration': final_iter,
                        
                        'eval/ngram_overlap_1': metrics['ngram_overlap_1'],
                        'eval/ngram_overlap_2': metrics['ngram_overlap_2'],
                        'eval/ngram_overlap_3': metrics['ngram_overlap_3'],
                        'eval/perplexity': metrics['perplexity'],
                        'eval/kl_divergence': metrics['kl_divergence'],
                        
                        'eval/self_bleu': metrics['self_bleu'],
                        'eval/distinct_1': metrics['distinct_1'],
                        'eval/distinct_2': metrics['distinct_2'],
                        'eval/distinct_3': metrics['distinct_3'],
                        'eval/entropy': metrics['entropy'],
                    })
                    
                    samples_path = os.path.join(out_dir, 'generated_samples.txt')
                    if os.path.exists(samples_path):
                        artifact = wandb.Artifact(
                            name=f'samples-{run_name}',
                            type='generated_text',
                            description=f'Generated samples from {run_name}'
                        )
                        artifact.add_file(samples_path)
                        wandb.log_artifact(artifact)
                    
                    wandb.finish()
                    print(f"✓ Metrics logged to WandB for {run_name}")
                    
                except Exception as e:
                    print(f"Warning: Failed to log metrics to WandB: {e}")
            
            print(f"\n✓ Configuration {i}/{len(CONFIGURATIONS)} complete!")
            
        except Exception as e:
            print(f"\n✗ Configuration {i}/{len(CONFIGURATIONS)} failed: {str(e)}")
            continue
    
    # Save results summary
    if all_results:
        csv_path = "results_extensive_search.csv"
        print(f"\n{'='*80}")
        print("SAVING RESULTS SUMMARY")
        print(f"{'='*80}\n")
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"Results saved to: {csv_path}")
        
        # Print summary table
        print(f"\n{'='*80}")
        print("RESULTS SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"{'Config':<20} {'Params':<10} {'Iters':<8} {'Time':<8} {'Val Loss':<10} {'Perplexity':<12}")
        print("-" * 90)
        
        for result in all_results:
            config = result['config_name']
            params = f"{result['total_params']/1e6:.1f}M"
            iters = f"{result['final_iteration']}" if isinstance(result['final_iteration'], int) else 'N/A'
            time_min = f"{result['training_time_min']:.1f}"
            val_loss = f"{result['val_loss']:.4f}" if isinstance(result['val_loss'], float) else str(result['val_loss'])
            perplexity = f"{result['perplexity']:.4f}" if 'perplexity' in result else 'N/A'
            
            print(f"{config:<20} {params:<10} {iters:<8} {time_min:<8} {val_loss:<10} {perplexity:<12}")
        
        # Find best configurations by different metrics
        valid_results = [r for r in all_results if isinstance(r['val_loss'], float)]
        if valid_results:
            best_val_loss = min(valid_results, key=lambda x: x['val_loss'])
            fastest = min(valid_results, key=lambda x: x['training_time_min'])
            
            print("\n" + "="*80)
            print("BEST CONFIGURATIONS")
            print("="*80)
            
            print("\nBest Validation Loss:")
            print(f"  Config:      {best_val_loss['config_name']} ({best_val_loss['description']})")
            print(f"  Val Loss:    {best_val_loss['val_loss']:.4f}")
            print(f"  Perplexity:  {best_val_loss['perplexity']:.4f}")
            print(f"  Iterations:  {best_val_loss['final_iteration']}")
            
            print("\nFastest Training:")
            print(f"  Config:      {fastest['config_name']} ({fastest['description']})")
            print(f"  Time:        {fastest['training_time_min']:.2f} min")
            print(f"  Val Loss:    {fastest['val_loss']:.4f}")
            print("="*80 + "\n")
    
    # Create WandB summary
    if WANDB_AVAILABLE and all_results:
        try:
            summary_run = wandb.init(
                project='shakespeare-extensive-search',
                name='00-summary-comparison',
                job_type='summary'
            )
            
            table_data = []
            for result in all_results:
                table_data.append([
                    result['config_name'],
                    result['n_layer'],
                    result['n_head'],
                    result['n_embd'],
                    result['total_params'],
                    result['final_iteration'],
                    result.get('val_loss', 'N/A'),
                    result.get('perplexity', 'N/A'),
                    result.get('ngram_overlap_2', 'N/A'),
                    result.get('kl_divergence', 'N/A'),
                    result.get('self_bleu', 'N/A'),
                    result.get('distinct_2', 'N/A'),
                    result.get('entropy', 'N/A'),
                    result['training_time_min'],
                ])
            
            table = wandb.Table(
                columns=[
                    'Config', 'Layers', 'Heads', 'Embd', 'Params', 'Iters',
                    'Val Loss', 'Perplexity', 'N-gram-2', 'KL Div',
                    'Self-BLEU', 'Distinct-2', 'Entropy', 'Time (min)'
                ],
                data=table_data
            )
            
            wandb.log({"results_comparison": table})
            
            if valid_results:
                wandb.log({
                    "best_config": best_val_loss['config_name'],
                    "best_val_loss": best_val_loss['val_loss'],
                    "best_perplexity": best_val_loss['perplexity'],
                })
            
            wandb.finish()
            print("✓ Summary table logged to WandB")
            
        except Exception as e:
            print(f"Warning: Failed to create WandB summary: {e}")
    
    print("\n" + "="*80)
    print("EXTENSIVE HYPERPARAMETER SEARCH COMPLETE!")
    print("="*80)
    print(f"Total configurations tested: {len(all_results)}")
    print(f"Results saved to: results_extensive_search.csv")
    print(f"Check WandB project: shakespeare-extensive-search")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

