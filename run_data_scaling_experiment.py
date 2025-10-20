"""
Data scaling experiment: Train nanoGPT on Linux kernel with varying dataset sizes.

Tests how model performance improves with more training data.
Uses fixed architecture (L4-H4-E256) - best from Part 1.
"""

import os
import sys
import time
import subprocess
import json
from typing import Dict, List, Tuple
import torch

from sample_and_evaluate import evaluate_model

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Metrics will not be logged to WandB.")


# Dataset sizes to test (matches prepare.py)
DATASET_SIZES = [
    ('100k', '100K characters'),
    ('500k', '500K characters'),
    ('1m', '1M characters'),
    ('5m', '5M characters'),
    ('10m', '10M characters'),
]

# Fixed architecture (best from Part 1)
ARCHITECTURE = {
    'n_layer': 4,
    'n_head': 4,
    'n_embd': 256,
}


def count_parameters(n_layer: int, n_head: int, n_embd: int, 
                     vocab_size: int = 100, block_size: int = 256) -> int:
    """Estimate the number of parameters in the model."""
    embedding_params = vocab_size * n_embd + block_size * n_embd
    attention_params = 4 * n_embd * n_embd
    mlp_params = 2 * n_embd * 4 * n_embd
    layernorm_params = 4 * n_embd
    block_params = (attention_params + mlp_params + layernorm_params) * n_layer
    output_params = n_embd + vocab_size * n_embd
    total_params = embedding_params + block_params + output_params
    return total_params


def train_on_dataset(
    dataset_name: str,
    dataset_desc: str,
    data_dir: str,
    base_config: str = "config/train_linux_kernel.py",
    num_gpus: int = 1
) -> Tuple[str, float, int, float, Dict]:
    """
    Train model on a specific dataset size.
    
    Returns:
        (out_dir, val_loss, final_iter, training_time, checkpoint_info)
    """
    n_layer = ARCHITECTURE['n_layer']
    n_head = ARCHITECTURE['n_head']
    n_embd = ARCHITECTURE['n_embd']
    
    # Create run name and output directory
    run_name = f"linux-{dataset_name}-L{n_layer}-H{n_head}-E{n_embd}"
    out_dir = f"/nobackup/gaurav/out-linux-{dataset_name}"
    
    # Ensure base directory exists
    os.makedirs('/nobackup/gaurav/', exist_ok=True)
    
    # Count parameters (approximate, actual vocab may vary)
    total_params = count_parameters(n_layer, n_head, n_embd, vocab_size=100)
    
    print(f"\n{'='*70}")
    print(f"Training on dataset: {dataset_name} ({dataset_desc})")
    print(f"Architecture: L{n_layer}-H{n_head}-E{n_embd} ({total_params:,} params)")
    print(f"Output: {out_dir}")
    print(f"{'='*70}\n")
    
    # Build training command
    if num_gpus > 1:
        # Multi-GPU training with DDP
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={num_gpus}",
            "train_with_early_stopping.py",
            base_config,
            f"--out_dir={out_dir}",
            f"--dataset={os.path.abspath(os.path.join(data_dir, dataset_name))}",
            f"--n_layer={n_layer}",
            f"--n_head={n_head}",
            f"--n_embd={n_embd}",
            f"--wandb_log=True",
            f"--wandb_project=linux-kernel-data-scaling",
            f"--wandb_run_name={run_name}",
        ]
    else:
        # Single GPU training
        cmd = [
            "python", "train_with_early_stopping.py",
            base_config,
            f"--out_dir={out_dir}",
            f"--dataset={os.path.abspath(os.path.join(data_dir, dataset_name))}",
            f"--n_layer={n_layer}",
            f"--n_head={n_head}",
            f"--n_embd={n_embd}",
            f"--wandb_log=True",
            f"--wandb_project=linux-kernel-data-scaling",
            f"--wandb_run_name={run_name}",
        ]
    
    print("Command:", " ".join(cmd))
    print()
    
    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        training_time = (time.time() - start_time) / 60  # minutes
        print(f"\n✓ Training completed in {training_time:.2f} minutes")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with exit code {e.returncode}")
        print("Check the output above for errors")
        return out_dir, float('inf'), 0, 0.0, {}
    
    # Give some time for processes to clean up
    print("Waiting for GPU cleanup...")
    time.sleep(5)
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Read checkpoint to get final metrics
    checkpoint_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found at {checkpoint_path}")
        return out_dir, float('inf'), 0, training_time, {}
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        val_loss = checkpoint.get('best_val_loss', float('inf'))
        final_iter = checkpoint.get('iter_num', 0)
        
        print(f"✓ Loaded checkpoint: val_loss={val_loss:.4f}, iter={final_iter}")
        
        return out_dir, val_loss, final_iter, training_time, checkpoint
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return out_dir, float('inf'), 0, training_time, {}


def run_data_scaling_experiment(num_gpus: int = 1, skip_existing: bool = True, data_dir: str = '/nobackup/gaurav/kernel_code'):
    """Run the complete data scaling experiment."""
    print("="*70)
    print("LINUX KERNEL DATA SCALING EXPERIMENT")
    print("="*70)
    print(f"Architecture: L{ARCHITECTURE['n_layer']}-H{ARCHITECTURE['n_head']}-E{ARCHITECTURE['n_embd']}")
    print(f"Datasets: {len(DATASET_SIZES)} sizes")
    print(f"Data directory: {data_dir}")
    print(f"GPUs: {num_gpus}")
    print(f"Skip existing: {skip_existing}")
    print("="*70)
    
    # Load existing results if available
    results = []
    completed_datasets = set()
    
    if os.path.exists('data_scaling_results.json') and skip_existing:
        print("\n✓ Found existing results file: data_scaling_results.json")
        with open('data_scaling_results.json', 'r') as f:
            results = json.load(f)
        
        completed_datasets = {r['dataset_name'] for r in results}
        
        if completed_datasets:
            print(f"✓ Already completed: {', '.join(sorted(completed_datasets))}")
            print(f"⚠ Will skip these and only process remaining datasets")
        
        response = input("\nContinue from where we left off? [Y/n]: ").strip().lower()
        if response == 'n':
            print("Starting fresh (existing results will be overwritten)...")
            results = []
            completed_datasets = set()
        else:
            print(f"Resuming... {len(completed_datasets)} already done, {len(DATASET_SIZES) - len(completed_datasets)} to go")
    
    for i, (dataset_name, dataset_desc) in enumerate(DATASET_SIZES, 1):
        print(f"\n\n{'#'*70}")
        print(f"# Dataset {i}/{len(DATASET_SIZES)}: {dataset_name}")
        print(f"{'#'*70}")
        
        # Skip if already completed
        if skip_existing and dataset_name in completed_datasets:
            print(f"✓ Already completed: {dataset_name}")
            print(f"  (Delete data_scaling_results.json or use --no-skip-existing to re-run)")
            continue
        
        # Check if dataset exists
        dataset_dir = os.path.join(data_dir, dataset_name)
        if not os.path.exists(dataset_dir):
            print(f"✗ Dataset not found at {dataset_dir}")
            print(f"  Please run: python data/linux_kernel/prepare.py --output_dir {data_dir}")
            continue
        
        # Check if model checkpoint already exists
        out_dir = f"/nobackup/gaurav/out-linux-{dataset_name}"
        checkpoint_path = os.path.join(out_dir, 'ckpt.pt')
        
        if skip_existing and os.path.exists(checkpoint_path):
            print(f"\n✓ Found existing checkpoint at {checkpoint_path}")
            response = input(f"Use existing checkpoint for {dataset_name}? [Y/n]: ").strip().lower()
            
            if response != 'n':
                print("Using existing checkpoint, skipping training...")
                
                # Load checkpoint to get metrics
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    val_loss = checkpoint.get('best_val_loss', float('inf'))
                    final_iter = checkpoint.get('iter_num', 0)
                    
                    print(f"✓ Checkpoint loaded: val_loss={val_loss:.4f}, iter={final_iter}")
                    
                    # Skip to evaluation
                    training_time = 0.0  # Unknown for existing checkpoint
                    
                except Exception as e:
                    print(f"✗ Error loading checkpoint: {e}")
                    print("Will re-train...")
                    checkpoint = None
                    val_loss = float('inf')
                    final_iter = 0
                    training_time = 0.0
            else:
                print("Re-training model...")
                checkpoint = None
        else:
            checkpoint = None
        
        # Train model (skip if checkpoint loaded above)
        if checkpoint is None:
            out_dir, val_loss, final_iter, training_time, checkpoint = train_on_dataset(
                dataset_name, dataset_desc, data_dir, num_gpus=num_gpus
            )
        else:
            # Already have checkpoint from above
            out_dir = f"/nobackup/gaurav/out-linux-{dataset_name}"
        
        if val_loss == float('inf'):
            print(f"✗ Skipping evaluation due to training failure")
            continue
        
        print(f"\n{'='*70}")
        print(f"EVALUATION: {dataset_name}")
        print(f"{'='*70}\n")
        
        # Evaluate model
        try:
            metrics = evaluate_model(
                out_dir=out_dir,
                num_samples=100,
                max_new_tokens=1000,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                save_samples=True
            )
            
            print(f"\n✓ Evaluation complete")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Perplexity: {metrics['perplexity']:.4f}")
            print(f"  KL Divergence: {metrics['kl_divergence']:.4f}")
            print(f"  Self-BLEU: {metrics['self_bleu']:.4f}")
            
        except Exception as e:
            print(f"✗ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            metrics = {}
        
        # Store results (convert any tensors to Python scalars for JSON serialization)
        def convert_to_serializable(obj):
            """Convert tensors and numpy types to Python scalars."""
            if torch.is_tensor(obj):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            else:
                return obj
        
        result = {
            'dataset_name': dataset_name,
            'dataset_desc': dataset_desc,
            'n_layer': ARCHITECTURE['n_layer'],
            'n_head': ARCHITECTURE['n_head'],
            'n_embd': ARCHITECTURE['n_embd'],
            'total_params': int(count_parameters(
                ARCHITECTURE['n_layer'], 
                ARCHITECTURE['n_head'], 
                ARCHITECTURE['n_embd']
            )),
            'out_dir': out_dir,
            'val_loss': float(val_loss),
            'final_iteration': int(final_iter),
            'training_time_min': float(training_time),
        }
        
        # Add metrics, converting any non-serializable types
        for key, value in metrics.items():
            result[key] = convert_to_serializable(value)
        
        results.append(result)
        
        # Log to WandB
        if WANDB_AVAILABLE and metrics:
            try:
                run_name = f"linux-{dataset_name}-L{ARCHITECTURE['n_layer']}-H{ARCHITECTURE['n_head']}-E{ARCHITECTURE['n_embd']}"
                wandb_settings = wandb.Settings(init_timeout=300)
                run = wandb.init(
                    project='linux-kernel-data-scaling', 
                    name=run_name, 
                    id=run_name, 
                    resume='allow',
                    settings=wandb_settings
                )
                
                # Log all metrics to summary
                wandb.summary['dataset/name'] = dataset_name
                wandb.summary['dataset/size'] = dataset_desc
                wandb.summary['config/n_layer'] = ARCHITECTURE['n_layer']
                wandb.summary['config/n_head'] = ARCHITECTURE['n_head']
                wandb.summary['config/n_embd'] = ARCHITECTURE['n_embd']
                wandb.summary['config/total_params'] = result['total_params']
                wandb.summary['config/training_time_min'] = training_time
                wandb.summary['config/final_iteration'] = final_iter
                wandb.summary['train/final_val_loss'] = val_loss
                
                for key, value in metrics.items():
                    wandb.summary[f'eval/{key}'] = value
                
                # Create visualization plots
                import matplotlib.pyplot as plt
                
                # Plot: Specific metrics
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                specific_metrics = {
                    'Unigram': metrics['ngram_overlap_1'],
                    'Bigram': metrics['ngram_overlap_2'],
                    'Trigram': metrics['ngram_overlap_3'],
                    'Perplexity': metrics['perplexity'],
                    'KL Div': metrics['kl_divergence']
                }
                bars1 = ax1.bar(specific_metrics.keys(), specific_metrics.values(), 
                               color=['#2E86AB', '#2E86AB', '#2E86AB', '#A23B72', '#F18F01'])
                ax1.set_title(f'Specific Metrics: {dataset_desc}\n{run_name}', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Score', fontsize=12)
                ax1.grid(axis='y', alpha=0.3)
                for bar in bars1:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=10)
                plt.tight_layout()
                wandb.log({"plots/specific_metrics": wandb.Image(fig1)})
                plt.close(fig1)
                
                # Plot: General metrics
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                general_metrics = {
                    'Self-BLEU': metrics['self_bleu'],
                    'Distinct-1': metrics['distinct_1'],
                    'Distinct-2': metrics['distinct_2'],
                    'Distinct-3': metrics['distinct_3'],
                    'Entropy': metrics['entropy']
                }
                bars2 = ax2.bar(general_metrics.keys(), general_metrics.values(), 
                               color=['#A23B72', '#06A77D', '#06A77D', '#06A77D', '#F18F01'])
                ax2.set_title(f'General Metrics: {dataset_desc}\n{run_name}', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Score', fontsize=12)
                ax2.grid(axis='y', alpha=0.3)
                for bar in bars2:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=10)
                plt.tight_layout()
                wandb.log({"plots/general_metrics": wandb.Image(fig2)})
                plt.close(fig2)
                
                wandb.finish()
                print(f"✓ Logged to WandB: {run_name}")
                
            except Exception as e:
                print(f"✗ WandB logging failed: {e}")
        
        # Save intermediate results
        with open('data_scaling_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Saved results to data_scaling_results.json")
    
    # Final summary
    print("\n\n" + "="*70)
    print("DATA SCALING EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nCompleted {len(results)}/{len(DATASET_SIZES)} experiments")
    print(f"\nResults summary:")
    print(f"{'Dataset':<10} {'Val Loss':<12} {'Perplexity':<12} {'Training Time':<15}")
    print("-" * 70)
    for r in results:
        print(f"{r['dataset_name']:<10} {r['val_loss']:<12.4f} {r.get('perplexity', 0):<12.2f} {r['training_time_min']:<15.2f} min")
    
    print(f"\nResults saved to: data_scaling_results.json")
    print(f"Next: python analyze_data_scaling.py")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run data scaling experiment on Linux kernel')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--data_dir', type=str, default='/nobackup/gaurav/kernel_code',
                       help='Directory containing prepared datasets (default: /nobackup/gaurav/kernel_code)')
    parser.add_argument('--no-skip-existing', action='store_true', 
                       help='Re-run all experiments even if results/checkpoints exist')
    args = parser.parse_args()
    
    results = run_data_scaling_experiment(
        num_gpus=args.num_gpus,
        skip_existing=not args.no_skip_existing,
        data_dir=args.data_dir
    )

