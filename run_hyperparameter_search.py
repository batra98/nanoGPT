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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Metrics will not be logged to WandB.")


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
        training_time = (time.time() - start_time) / 60
        
    except subprocess.CalledProcessError:
        raise
    
    return out_dir, training_time, {}


def main():
    """Run hyperparameter search."""
    import argparse
    parser = argparse.ArgumentParser(description='Run hyperparameter search')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to use per configuration (1 or 8)')
    parser.add_argument('--skip_eval', action='store_true',
                        help='Skip evaluation phase (sample generation and metrics)')
    args = parser.parse_args()
    
    num_gpus = args.num_gpus
    
    print(f"\nStarting search: {len(CONFIGURATIONS)} configs, {num_gpus} GPUs each\n")
    
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
            
            # Cleanup GPU and wait for processes
            if num_gpus > 1:
                print("Cleaning up DDP processes...")
                time.sleep(5)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                time.sleep(1)
            
            # Optionally evaluate on single GPU
            if args.skip_eval:
                print("Skipping evaluation (--skip_eval enabled)")
                metrics = {}  # Empty metrics dict
            else:
                print("Starting evaluation...")
                eval_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                try:
                    metrics = evaluate_model(out_dir, 50, 500, eval_device, True)
                except Exception as eval_error:
                    print(f"Evaluation failed: {eval_error}")
                    raise
            
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
            
            # Log to WandB (with error handling)
            if WANDB_AVAILABLE and not args.skip_eval:
                try:
                    run_name = f"shakespeare-L{n_layer}-H{n_head}-E{n_embd}"
                    run = wandb.init(project='shakespeare-hyperparam-search', name=run_name, id=run_name, resume='allow')
                    
                    # Log config and metrics (if available)
                    log_dict = {
                        'config/n_layer': n_layer, 'config/n_head': n_head, 'config/n_embd': n_embd,
                        'config/total_params': result['total_params'], 'config/training_time_min': training_time,
                    }
                    
                    # Add evaluation metrics if they exist
                    if metrics:
                        log_dict.update({
                            'eval/ngram_overlap_1': metrics['ngram_overlap_1'], 'eval/ngram_overlap_2': metrics['ngram_overlap_2'],
                            'eval/ngram_overlap_3': metrics['ngram_overlap_3'], 'eval/perplexity': metrics['perplexity'],
                            'eval/kl_divergence': metrics['kl_divergence'], 'eval/self_bleu': metrics['self_bleu'],
                            'eval/distinct_1': metrics['distinct_1'], 'eval/distinct_2': metrics['distinct_2'],
                            'eval/distinct_3': metrics['distinct_3'], 'eval/entropy': metrics['entropy'],
                        })
                    
                    wandb.log(log_dict)
                    
                    if os.path.exists(os.path.join(out_dir, 'generated_samples.txt')):
                        artifact = wandb.Artifact(f'samples-{run_name}', 'generated_text')
                        artifact.add_file(os.path.join(out_dir, 'generated_samples.txt'))
                        wandb.log_artifact(artifact)
                    wandb.finish()
                except Exception as wandb_error:
                    print(f"WandB logging failed: {wandb_error}")
            
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
            
            summary_run = wandb.init(project='shakespeare-hyperparam-search', name='00-summary', job_type='summary')
            table_data = [[r['config_name'], r['n_layer'], r['n_head'], r['n_embd'], r['total_params'],
                           r.get('val_loss', 'N/A'), r.get('perplexity', 'N/A'),
                           r.get('kl_divergence', 'N/A'), r.get('self_bleu', 'N/A'), r['training_time_min']]
                          for r in all_results]
            table = wandb.Table(columns=['Config', 'Layers', 'Heads', 'Embd', 'Params',
                                         'Val Loss', 'Perplexity', 'KL Div', 'Self-BLEU', 'Time'],
                               data=table_data)
            wandb.log({"results": table})
            if best:
                wandb.log({"best_config": best['config_name'], "best_val_loss": best['val_loss']})
            wandb.finish()
        except Exception as summary_error:
            print(f"Summary table creation failed: {summary_error}")
    
    print(f"\nComplete: {len(all_results)} configs. Check WandB: shakespeare-hyperparam-search\n")


if __name__ == '__main__':
    main()

