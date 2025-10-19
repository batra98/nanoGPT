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
            text=True,
            timeout=None  # No timeout, but process must exit cleanly
        )
        training_time = (time.time() - start_time) / 60
        
    except subprocess.CalledProcessError:
        raise
    except subprocess.TimeoutExpired:
        print("Training process timed out!")
        raise
    
    return out_dir, training_time, {}


def main():
    """Run extensive hyperparameter search with early stopping."""
    import argparse
    parser = argparse.ArgumentParser(description='Run extensive hyperparameter search')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to use per configuration (1 or 8)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to generate for evaluation')
    args = parser.parse_args()
    
    num_gpus = args.num_gpus
    
    print(f"\nStarting extensive search: {len(CONFIGURATIONS)} configs, {num_gpus} GPUs each\n")
    
    # Check data and device
    if not os.path.exists("data/shakespeare_char/train.bin"):
        print("ERROR: Run python data/shakespeare_char/prepare.py first")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if num_gpus > 1 and torch.cuda.is_available():
        num_gpus = min(num_gpus, torch.cuda.device_count())
    
    # Results storage
    all_results = []
    
    # Train each configuration
    for i, (n_layer, n_head, n_embd, description) in enumerate(CONFIGURATIONS, 1):
        print(f"\n[{i}/{len(CONFIGURATIONS)}] L{n_layer}-H{n_head}-E{n_embd}")
        
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
            
            # Cleanup GPU - wait for processes to exit naturally first
            if num_gpus > 1:
                print("Waiting for DDP processes to exit...")
                time.sleep(5)  # Give processes time to clean up naturally
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                time.sleep(1)
            
            # Always evaluate on single GPU
            print("Starting evaluation...")
            eval_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            try:
                metrics = evaluate_model(out_dir, args.num_samples, 500, eval_device, True)
            except Exception as eval_error:
                print(f"Evaluation failed: {eval_error}")
                raise
            
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
            
            # Always log to WandB
            if WANDB_AVAILABLE:
                try:
                    run_name = f"shakespeare-L{n_layer}-H{n_head}-E{n_embd}"
                    # Increase timeout and add retry logic
                    wandb_settings = wandb.Settings(init_timeout=300)  # 5 minutes
                    run = wandb.init(
                        project='shakespeare-extensive-search', 
                        name=run_name, 
                        id=run_name, 
                        resume='allow',
                        settings=wandb_settings
                    )
                    
                    # Log everything: config, training, and evaluation metrics
                    log_dict = {
                        # Config
                        'config/n_layer': n_layer,
                        'config/n_head': n_head,
                        'config/n_embd': n_embd,
                        'config/total_params': result['total_params'],
                        'config/training_time_min': training_time,
                        'config/final_iteration': final_iter,
                        
                        # Training metrics
                        'train/final_val_loss': val_loss,
                        
                        # Evaluation metrics (always available)
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
                    }
                    
                    wandb.log(log_dict)
                    
                    # Create visualization plots for metrics
                    import matplotlib.pyplot as plt
                    import numpy as np
                    
                    # Plot 1: Specific metrics (distribution closeness) - lower is better
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    specific_metrics = {
                        'Unigram\nOverlap': metrics['ngram_overlap_1'],
                        'Bigram\nOverlap': metrics['ngram_overlap_2'],
                        'Trigram\nOverlap': metrics['ngram_overlap_3'],
                        'Perplexity': metrics['perplexity'],
                        'KL Div': metrics['kl_divergence']
                    }
                    bars1 = ax1.bar(specific_metrics.keys(), specific_metrics.values(), color=['#2E86AB', '#2E86AB', '#2E86AB', '#A23B72', '#F18F01'])
                    ax1.set_title(f'Specific Metrics: Distribution Closeness\n{run_name}', fontsize=14, fontweight='bold')
                    ax1.set_ylabel('Score', fontsize=12)
                    ax1.grid(axis='y', alpha=0.3)
                    for bar in bars1:
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
                    plt.tight_layout()
                    wandb.log({"plots/specific_metrics": wandb.Image(fig1)})
                    plt.close(fig1)
                    
                    # Plot 2: General metrics (quality/diversity) - higher is better (except self-BLEU)
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    general_metrics = {
                        'Self-BLEU\n(lower=diverse)': metrics['self_bleu'],
                        'Distinct-1': metrics['distinct_1'],
                        'Distinct-2': metrics['distinct_2'],
                        'Distinct-3': metrics['distinct_3'],
                        'Entropy': metrics['entropy']
                    }
                    bars2 = ax2.bar(general_metrics.keys(), general_metrics.values(), color=['#A23B72', '#06A77D', '#06A77D', '#06A77D', '#F18F01'])
                    ax2.set_title(f'General Metrics: Quality & Diversity\n{run_name}', fontsize=14, fontweight='bold')
                    ax2.set_ylabel('Score', fontsize=12)
                    ax2.grid(axis='y', alpha=0.3)
                    for bar in bars2:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
                    plt.tight_layout()
                    wandb.log({"plots/general_metrics": wandb.Image(fig2)})
                    plt.close(fig2)
                    
                    # Plot 3: Combined summary with normalized scores
                    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))
                    
                    # N-gram overlap progression
                    ngram_data = [metrics['ngram_overlap_1'], metrics['ngram_overlap_2'], metrics['ngram_overlap_3']]
                    ax3a.plot([1, 2, 3], ngram_data, marker='o', linewidth=2, markersize=8, color='#2E86AB')
                    ax3a.fill_between([1, 2, 3], ngram_data, alpha=0.3, color='#2E86AB')
                    ax3a.set_xlabel('N-gram Size', fontsize=12)
                    ax3a.set_ylabel('Overlap %', fontsize=12)
                    ax3a.set_title('N-gram Overlap Progression', fontsize=12, fontweight='bold')
                    ax3a.set_xticks([1, 2, 3])
                    ax3a.grid(True, alpha=0.3)
                    for i, val in enumerate(ngram_data, 1):
                        ax3a.text(i, val, f'{val:.2f}%', ha='center', va='bottom')
                    
                    # Distinct-n progression
                    distinct_data = [metrics['distinct_1'], metrics['distinct_2'], metrics['distinct_3']]
                    ax3b.plot([1, 2, 3], distinct_data, marker='s', linewidth=2, markersize=8, color='#06A77D')
                    ax3b.fill_between([1, 2, 3], distinct_data, alpha=0.3, color='#06A77D')
                    ax3b.set_xlabel('N-gram Size', fontsize=12)
                    ax3b.set_ylabel('Distinct Ratio', fontsize=12)
                    ax3b.set_title('Distinct-N Diversity', fontsize=12, fontweight='bold')
                    ax3b.set_xticks([1, 2, 3])
                    ax3b.grid(True, alpha=0.3)
                    for i, val in enumerate(distinct_data, 1):
                        ax3b.text(i, val, f'{val:.3f}', ha='center', va='bottom')
                    
                    plt.suptitle(f'Metric Progressions: {run_name}', fontsize=14, fontweight='bold', y=1.02)
                    plt.tight_layout()
                    wandb.log({"plots/metric_progressions": wandb.Image(fig3)})
                    plt.close(fig3)
                    
                    try:
                        wandb.finish()
                    except:
                        pass  # Ignore finish errors
                except Exception as wandb_error:
                    print(f"⚠ WandB logging failed: {wandb_error}")
                    print("Continuing without WandB logging...")
            
            print(f"✓ Config {i}/{len(CONFIGURATIONS)} complete")
            
        except Exception as e:
            print(f"✗ Config {i}/{len(CONFIGURATIONS)} failed: {e}")
            # Force cleanup before continuing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import traceback
            traceback.print_exc()
            continue
    
    # WandB summary table (with error handling)
    if WANDB_AVAILABLE and all_results:
        try:
            valid_results = [r for r in all_results if isinstance(r['val_loss'], float)]
            best = min(valid_results, key=lambda x: x['val_loss']) if valid_results else None
            
            wandb_settings = wandb.Settings(init_timeout=300)
            summary_run = wandb.init(
                project='shakespeare-extensive-search', 
                name='00-summary', 
                job_type='summary',
                settings=wandb_settings
            )
            table_data = [[r['config_name'], r['n_layer'], r['n_head'], r['n_embd'], r['total_params'],
                           r['final_iteration'], r.get('val_loss', 'N/A'), r.get('perplexity', 'N/A'),
                           r.get('kl_divergence', 'N/A'), r.get('self_bleu', 'N/A'), r['training_time_min']]
                          for r in all_results]
            table = wandb.Table(columns=['Config', 'Layers', 'Heads', 'Embd', 'Params', 'Iters',
                                         'Val Loss', 'Perplexity', 'KL Div', 'Self-BLEU', 'Time'],
                               data=table_data)
            wandb.log({"results": table})
            if best:
                wandb.log({"best_config": best['config_name'], "best_val_loss": best['val_loss']})
            wandb.finish()
        except Exception as summary_error:
            print(f"Summary table creation failed: {summary_error}")
    
    print(f"\nComplete: {len(all_results)} configs. Check WandB: shakespeare-extensive-search\n")


if __name__ == '__main__':
    main()

