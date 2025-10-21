"""
Orchestration script for fine-tuning Shakespeare model on Linux kernel code.

This script runs a series of fine-tuning experiments with varying dataset sizes
and training iterations to study the transition from Shakespearean to C code.
"""

import os
import sys
import json
import shutil
import subprocess
import argparse
import time
import pickle
import numpy as np
from pathlib import Path

import torch
import wandb

from evaluation_metrics import EvaluationMetrics
from domain_transition_metrics import DomainTransitionMetrics
from sample_and_evaluate import load_model, generate_samples


def download_pretrained_checkpoint(output_dir='out-shakespeare-pretrained'):
    """
    Instructions and helper to download the pre-trained Shakespeare checkpoint.
    
    The checkpoint should be downloaded from:
    https://huggingface.co/batra98/hyperparameter-tuning-shakespeare/tree/main/out-shakespeare-extensive-L4-H4-E256
    
    Args:
        output_dir: Directory to save the checkpoint
    """
    ckpt_path = os.path.join(output_dir, 'ckpt.pt')
    
    if os.path.exists(ckpt_path):
        print(f"✓ Found pre-trained checkpoint at {ckpt_path}")
        return ckpt_path
    
    print(f"\n{'='*70}")
    print("PRE-TRAINED CHECKPOINT REQUIRED")
    print(f"{'='*70}")
    print("\nPlease download the Shakespeare checkpoint:")
    print("1. Visit: https://huggingface.co/batra98/hyperparameter-tuning-shakespeare")
    print("2. Navigate to: out-shakespeare-extensive-L4-H4-E256/")
    print("3. Download: ckpt.pt")
    print(f"4. Save to: {os.path.abspath(ckpt_path)}")
    print("\nAlternatively, use wget:")
    print(f"  wget https://huggingface.co/batra98/hyperparameter-tuning-shakespeare/resolve/main/out-shakespeare-extensive-L4-H4-E256/ckpt.pt -O {ckpt_path}")
    print(f"{'='*70}\n")
    
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Please download it first.")


def load_reference_texts(shakespeare_data_dir='data/shakespeare_char', 
                         kernel_data_dir='/nobackup/gaurav/kernel_code/5m'):
    """
    Load reference texts for computing transition metrics.
    
    Args:
        shakespeare_data_dir: Shakespeare dataset directory
        kernel_data_dir: Kernel code dataset directory
    
    Returns:
        Tuple of (shakespeare_text, kernel_text)
    """
    print("\nLoading reference texts for transition metrics...")
    
    # Load Shakespeare reference
    shakespeare_train = os.path.join(shakespeare_data_dir, 'train.bin')
    shakespeare_meta = os.path.join(shakespeare_data_dir, 'meta.pkl')
    
    if os.path.exists(shakespeare_train) and os.path.exists(shakespeare_meta):
        with open(shakespeare_meta, 'rb') as f:
            meta = pickle.load(f)
        itos = meta['itos']
        
        data = np.memmap(shakespeare_train, dtype=np.uint16, mode='r')
        # Sample 100k characters
        sample_size = min(100000, len(data))
        shakespeare_text = ''.join([itos[i] for i in data[:sample_size]])
        print(f"  ✓ Loaded {len(shakespeare_text)} chars from Shakespeare")
    else:
        print(f"  ⚠ Shakespeare reference not found at {shakespeare_data_dir}")
        shakespeare_text = None
    
    # Load kernel reference
    kernel_train = os.path.join(kernel_data_dir, 'train.bin')
    kernel_meta = os.path.join(kernel_data_dir, 'meta.pkl')
    
    if os.path.exists(kernel_train) and os.path.exists(kernel_meta):
        with open(kernel_meta, 'rb') as f:
            meta = pickle.load(f)
        itos = meta['itos']
        
        data = np.memmap(kernel_train, dtype=np.uint16, mode='r')
        # Sample 100k characters
        sample_size = min(100000, len(data))
        kernel_text = ''.join([itos[i] for i in data[:sample_size]])
        print(f"  ✓ Loaded {len(kernel_text)} chars from kernel code")
    else:
        print(f"  ⚠ Kernel reference not found at {kernel_data_dir}")
        kernel_text = None
    
    return shakespeare_text, kernel_text


def finetune_model(base_checkpoint_dir, dataset_path, max_iters, num_gpus=8):
    """
    Fine-tune the model on a specific dataset for a given number of iterations.
    
    Args:
        base_checkpoint_dir: Directory containing the base checkpoint
        dataset_path: Path to the dataset to fine-tune on
        max_iters: Number of training iterations
        num_gpus: Number of GPUs to use
    
    Returns:
        Path to the fine-tuned checkpoint directory
    """
    # Create output directory for this run
    dataset_name = Path(dataset_path).name
    out_dir = f'out-finetune-{dataset_name}-{max_iters}iter'
    
    print(f"\n{'='*70}")
    print(f"FINE-TUNING: {dataset_name} for {max_iters} iterations")
    print(f"{'='*70}")
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Verify base checkpoint exists
    base_ckpt = os.path.join(base_checkpoint_dir, 'ckpt.pt')
    if not os.path.exists(base_ckpt):
        raise FileNotFoundError(f"Base checkpoint not found at {base_ckpt}")
    
    # Prepare training command using transfer learning script
    dataset_abs_path = os.path.abspath(dataset_path)
    
    cmd = [
        'torchrun',
        f'--nproc_per_node={num_gpus}',
        'train_transfer_learning.py',
        'config/finetune_kernel.py',
        f'--out_dir={out_dir}',
        f'--dataset={dataset_abs_path}',
        f'--max_iters={max_iters}',
        f'--lr_decay_iters={max_iters}',
        f'--wandb_run_name=finetune-{dataset_name}-{max_iters}iter',
        f'--source_checkpoint={os.path.abspath(base_checkpoint_dir)}/ckpt.pt',
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    
    # Run training
    result = subprocess.run(cmd, cwd=os.getcwd())
    
    if result.returncode != 0:
        print(f"⚠ Warning: Training exited with code {result.returncode}")
    
    return out_dir


def evaluate_finetuned_model(out_dir, num_samples=10, max_new_tokens=500,
                             shakespeare_ref=None, kernel_ref=None):
    """
    Evaluate a fine-tuned model: generate samples and compute all metrics.
    
    Args:
        out_dir: Directory containing the model checkpoint
        num_samples: Number of samples to generate
        max_new_tokens: Tokens per sample
        shakespeare_ref: Reference Shakespeare text for transition metrics
        kernel_ref: Reference kernel text for transition metrics
    
    Returns:
        Dictionary of all metrics
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING: {out_dir}")
    print(f"{'='*70}")
    
    # Check if checkpoint exists
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        print(f"⚠ No checkpoint found at {ckpt_path}")
        return None
    
    # Load model
    print("\n1. Loading model...")
    try:
        model, encode, decode = load_model(out_dir, device='cuda')
        
        # Get validation loss from checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        val_loss = checkpoint.get('best_val_loss', 0.0)
        
    except Exception as e:
        print(f"⚠ Error loading model: {e}")
        return None
    
    # Get dataset info
    config = checkpoint.get('config', {})
    dataset = config.get('dataset', 'shakespeare_char')
    
    if os.path.isabs(dataset):
        data_dir = dataset
    else:
        data_dir = os.path.join('data', dataset)
    
    # Check if samples already exist
    samples_path = os.path.join(out_dir, 'generated_samples.txt')
    
    if os.path.exists(samples_path):
        print(f"\n2. Loading existing samples from {samples_path}...")
        with open(samples_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Parse samples
        generated_samples = []
        lines = content.split('\n')
        current_sample = []
        in_sample = False
        
        for line in lines:
            if line.startswith('='*60):
                if current_sample and in_sample:
                    generated_samples.append('\n'.join(current_sample))
                    current_sample = []
                    in_sample = False
            elif line.startswith('Sample '):
                in_sample = True
            elif in_sample:
                current_sample.append(line)
        
        if current_sample:
            generated_samples.append('\n'.join(current_sample))
        
        print(f"  ✓ Loaded {len(generated_samples)} existing samples")
    else:
        # Generate new samples
        print(f"\n2. Generating {num_samples} samples...")
        generated_samples = generate_samples(
            model=model,
            encode=encode,
            decode=decode,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            device='cuda'
        )
        
        # Save samples
        with open(samples_path, 'w', encoding='utf-8') as f:
            for i, sample in enumerate(generated_samples):
                f.write(f"{'='*60}\n")
                f.write(f"Sample {i+1}\n")
                f.write(f"{'='*60}\n")
                f.write(sample)
                f.write(f"\n\n")
        print(f"  ✓ Saved samples to {samples_path}")
    
    # Compute standard metrics
    print("\n3. Computing standard evaluation metrics...")
    evaluator = EvaluationMetrics(data_dir=data_dir)
    standard_metrics = evaluator.compute_all_metrics(generated_samples, val_loss)
    
    # Compute transition metrics
    print("\n4. Computing domain transition metrics...")
    transition_evaluator = DomainTransitionMetrics()
    transition_metrics = transition_evaluator.compute_all_transition_metrics(
        generated_samples,
        kernel_reference=kernel_ref,
        shakespeare_reference=shakespeare_ref
    )
    
    # Combine all metrics
    all_metrics = {**standard_metrics, **transition_metrics}
    
    # Save metrics
    metrics_path = os.path.join(out_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  ✓ Saved metrics to {metrics_path}")
    
    # Print key transition metrics
    print(f"\n{'='*70}")
    print("KEY TRANSITION METRICS")
    print(f"{'='*70}")
    print(f"  Transition score:        {transition_metrics['transition_score']:.3f} (0=Shakespeare, 1=C code)")
    print(f"  Code likeness:           {transition_metrics['code_likeness_score']:.3f}")
    print(f"  Shakespeare likeness:    {transition_metrics['shakespeare_likeness_score']:.3f}")
    print(f"  C keyword freq:          {transition_metrics['c_keyword_freq']:.2f} per 1k chars")
    print(f"  Shakespeare word freq:   {transition_metrics['shakespeare_word_freq']:.2f} per 1k chars")
    print(f"  Semicolon density:       {transition_metrics['semicolon_density']:.2f} per 1k chars")
    print(f"{'='*70}\n")
    
    return all_metrics


def run_experiments(base_checkpoint_dir, data_base_dir, num_gpus=8, 
                   data_sizes=None, iteration_counts=None):
    """
    Run the full fine-tuning experiment suite.
    
    Args:
        base_checkpoint_dir: Directory containing pre-trained Shakespeare model
        data_base_dir: Base directory for kernel code datasets
        num_gpus: Number of GPUs to use
        data_sizes: List of data sizes to experiment with
        iteration_counts: List of iteration counts to try
    
    Returns:
        Dictionary of all experiment results
    """
    if data_sizes is None:
        data_sizes = ['100k', '500k', '1m', '5m']
    
    if iteration_counts is None:
        iteration_counts = [100, 250, 500, 1000, 2000]
    
    # Load reference texts once
    shakespeare_ref, kernel_ref = load_reference_texts(
        shakespeare_data_dir='data/shakespeare_char',
        kernel_data_dir=os.path.join(data_base_dir, '5m')
    )
    
    # Results storage
    results = {
        'experiments': [],
        'shakespeare_reference': shakespeare_ref[:1000] if shakespeare_ref else None,
        'kernel_reference': kernel_ref[:1000] if kernel_ref else None,
    }
    
    results_file = 'finetuning_experiment_results.json'
    
    # Load existing results if available
    if os.path.exists(results_file):
        print(f"\nLoading existing results from {results_file}...")
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except json.JSONDecodeError:
            print("⚠ Could not load existing results, starting fresh")
    
    total_experiments = len(data_sizes) * len(iteration_counts)
    current_experiment = 0
    
    print(f"\n{'='*70}")
    print(f"STARTING FINE-TUNING EXPERIMENTS")
    print(f"{'='*70}")
    print(f"Data sizes: {data_sizes}")
    print(f"Iteration counts: {iteration_counts}")
    print(f"Total experiments: {total_experiments}")
    print(f"{'='*70}\n")
    
    for data_size in data_sizes:
        dataset_path = os.path.join(data_base_dir, data_size)
        
        if not os.path.exists(dataset_path):
            print(f"⚠ Skipping {data_size}: dataset not found at {dataset_path}")
            continue
        
        for max_iters in iteration_counts:
            current_experiment += 1
            print(f"\n{'#'*70}")
            print(f"EXPERIMENT {current_experiment}/{total_experiments}")
            print(f"Dataset: {data_size}, Iterations: {max_iters}")
            print(f"{'#'*70}")
            
            # Check if already completed
            exp_key = f"{data_size}-{max_iters}iter"
            existing = [e for e in results['experiments'] if e.get('exp_id') == exp_key]
            if existing and existing[0].get('metrics'):
                print(f"✓ Already completed, skipping...")
                continue
            
            try:
                # Fine-tune model
                out_dir = finetune_model(
                    base_checkpoint_dir=base_checkpoint_dir,
                    dataset_path=dataset_path,
                    max_iters=max_iters,
                    num_gpus=num_gpus
                )
                
                # Evaluate
                metrics = evaluate_finetuned_model(
                    out_dir=out_dir,
                    num_samples=10,
                    max_new_tokens=500,
                    shakespeare_ref=shakespeare_ref,
                    kernel_ref=kernel_ref
                )
                
                # Store results
                experiment_result = {
                    'exp_id': exp_key,
                    'data_size': data_size,
                    'max_iters': max_iters,
                    'out_dir': out_dir,
                    'metrics': metrics,
                    'timestamp': time.time()
                }
                
                # Update results
                results['experiments'] = [e for e in results['experiments'] if e.get('exp_id') != exp_key]
                results['experiments'].append(experiment_result)
                
                # Save after each experiment
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                print(f"\n✓ Experiment {current_experiment}/{total_experiments} complete")
                
            except Exception as e:
                print(f"\n⚠ Experiment failed: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {results_file}")
    print(f"Total experiments: {len(results['experiments'])}")
    print(f"{'='*70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run fine-tuning experiments')
    parser.add_argument('--base_checkpoint_dir', type=str, default='out-shakespeare-pretrained',
                       help='Directory containing pre-trained Shakespeare checkpoint')
    parser.add_argument('--data_base_dir', type=str, default='/nobackup/gaurav/kernel_code',
                       help='Base directory for kernel code datasets')
    parser.add_argument('--num_gpus', type=int, default=8,
                       help='Number of GPUs to use')
    parser.add_argument('--data_sizes', type=str, nargs='+', default=['100k', '500k', '1m', '5m'],
                       help='Data sizes to experiment with')
    parser.add_argument('--iterations', type=int, nargs='+', default=[100, 250, 500, 1000, 2000],
                       help='Iteration counts to try')
    
    args = parser.parse_args()
    
    # Check for pre-trained checkpoint
    download_pretrained_checkpoint(args.base_checkpoint_dir)
    
    # Run experiments
    results = run_experiments(
        base_checkpoint_dir=args.base_checkpoint_dir,
        data_base_dir=args.data_base_dir,
        num_gpus=args.num_gpus,
        data_sizes=args.data_sizes,
        iteration_counts=args.iterations
    )
    
    print("\n✓ All experiments complete!")
    print(f"Run 'python analyze_finetuning.py' to generate visualizations and report.")


if __name__ == '__main__':
    main()

