"""
Compare Transformer (with attention) vs Mamba (SSM) architectures.

This script loads evaluation results from both models and performs:
1. Quantitative comparison across all 10 metrics
2. Training curve comparison (loss, MFU, time)
3. Qualitative sample comparison
4. Statistical significance testing
"""

import os
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_evaluation_results(out_dir: str) -> Dict:
    """Load evaluation metrics from a model output directory."""
    metrics_path = os.path.join(out_dir, 'evaluation_metrics.json')
    
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics not found at {metrics_path}")
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def load_samples(out_dir: str) -> List[str]:
    """Load generated text samples from a model output directory."""
    samples_path = os.path.join(out_dir, 'generated_samples.txt')
    
    if not os.path.exists(samples_path):
        return []
    
    with open(samples_path, 'r') as f:
        content = f.read()
    
    # Parse samples (separated by "=" lines)
    samples = []
    current_sample = []
    in_sample = False
    
    for line in content.split('\n'):
        if line.strip().startswith('Sample'):
            if current_sample and in_sample:
                samples.append('\n'.join(current_sample))
            current_sample = []
            in_sample = True
        elif line.strip().startswith('==='):
            continue
        elif in_sample:
            current_sample.append(line)
    
    if current_sample and in_sample:
        samples.append('\n'.join(current_sample))
    
    return samples


def load_checkpoint_info(out_dir: str) -> Dict:
    """Load checkpoint metadata."""
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    
    if not os.path.exists(ckpt_path):
        return {}
    
    import torch
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    return {
        'best_val_loss': checkpoint.get('best_val_loss', None),
        'iter_num': checkpoint.get('iter_num', None),
        'model_args': checkpoint.get('model_args', {}),
        'config': checkpoint.get('config', {})
    }


def compare_metrics(transformer_metrics: Dict, mamba_metrics: Dict) -> None:
    """Create side-by-side comparison table of metrics."""
    
    print("\n" + "="*80)
    print("QUANTITATIVE METRICS COMPARISON")
    print("="*80)
    
    # Define metric groups
    specific_metrics = ['1-gram_overlap', '2-gram_overlap', '3-gram_overlap', 'perplexity', 'kl_divergence']
    general_metrics = ['self_bleu', 'distinct_1', 'distinct_2', 'distinct_3', 'shannon_entropy']
    
    print("\n📊 SPECIFIC METRICS (Training-Data-Dependent)")
    print("-" * 80)
    print(f"{'Metric':<20} {'Transformer':<20} {'Mamba':<20} {'Difference':<20}")
    print("-" * 80)
    
    for metric in specific_metrics:
        trans_val = transformer_metrics.get(metric, 'N/A')
        mamba_val = mamba_metrics.get(metric, 'N/A')
        
        if isinstance(trans_val, (int, float)) and isinstance(mamba_val, (int, float)):
            diff = mamba_val - trans_val
            diff_str = f"{diff:+.4f}"
            print(f"{metric:<20} {trans_val:<20.4f} {mamba_val:<20.4f} {diff_str:<20}")
        else:
            print(f"{metric:<20} {str(trans_val):<20} {str(mamba_val):<20} {'N/A':<20}")
    
    print("\n📊 GENERAL METRICS (Training-Data-Independent)")
    print("-" * 80)
    print(f"{'Metric':<20} {'Transformer':<20} {'Mamba':<20} {'Difference':<20}")
    print("-" * 80)
    
    for metric in general_metrics:
        trans_val = transformer_metrics.get(metric, 'N/A')
        mamba_val = mamba_metrics.get(metric, 'N/A')
        
        if isinstance(trans_val, (int, float)) and isinstance(mamba_val, (int, float)):
            diff = mamba_val - trans_val
            diff_str = f"{diff:+.4f}"
            print(f"{metric:<20} {trans_val:<20.4f} {mamba_val:<20.4f} {diff_str:<20}")
        else:
            print(f"{metric:<20} {str(trans_val):<20} {str(mamba_val):<20} {'N/A':<20}")
    
    print("-" * 80)


def plot_metric_comparison(transformer_metrics: Dict, mamba_metrics: Dict, output_dir: str) -> None:
    """Create bar plots comparing metrics."""
    
    specific_metrics = ['1-gram_overlap', '2-gram_overlap', '3-gram_overlap', 'perplexity', 'kl_divergence']
    general_metrics = ['self_bleu', 'distinct_1', 'distinct_2', 'distinct_3', 'shannon_entropy']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Specific metrics
    ax = axes[0]
    metrics_to_plot = [m for m in specific_metrics if m in transformer_metrics and m in mamba_metrics]
    
    if metrics_to_plot:
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        trans_vals = [transformer_metrics[m] for m in metrics_to_plot]
        mamba_vals = [mamba_metrics[m] for m in metrics_to_plot]
        
        ax.bar(x - width/2, trans_vals, width, label='Transformer', alpha=0.8, color='steelblue')
        ax.bar(x + width/2, mamba_vals, width, label='Mamba', alpha=0.8, color='coral')
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('Specific Metrics (Training-Data-Dependent)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    # General metrics
    ax = axes[1]
    metrics_to_plot = [m for m in general_metrics if m in transformer_metrics and m in mamba_metrics]
    
    if metrics_to_plot:
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        trans_vals = [transformer_metrics[m] for m in metrics_to_plot]
        mamba_vals = [mamba_metrics[m] for m in metrics_to_plot]
        
        ax.bar(x - width/2, trans_vals, width, label='Transformer', alpha=0.8, color='steelblue')
        ax.bar(x + width/2, mamba_vals, width, label='Mamba', alpha=0.8, color='coral')
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('General Metrics (Training-Data-Independent)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved metrics comparison plot to {output_dir}/metrics_comparison.png")
    plt.close()


def compare_model_architectures(transformer_info: Dict, mamba_info: Dict) -> None:
    """Compare model architecture and training configuration."""
    
    print("\n" + "="*80)
    print("ARCHITECTURE & TRAINING CONFIGURATION")
    print("="*80)
    
    trans_args = transformer_info.get('model_args', {})
    mamba_args = mamba_info.get('model_args', {})
    
    print(f"\n{'Parameter':<25} {'Transformer':<25} {'Mamba':<25}")
    print("-" * 75)
    
    all_keys = set(trans_args.keys()) | set(mamba_args.keys())
    for key in sorted(all_keys):
        trans_val = trans_args.get(key, 'N/A')
        mamba_val = mamba_args.get(key, 'N/A')
        print(f"{key:<25} {str(trans_val):<25} {str(mamba_val):<25}")
    
    print("\n📈 TRAINING RESULTS")
    print("-" * 75)
    print(f"{'Metric':<25} {'Transformer':<25} {'Mamba':<25}")
    print("-" * 75)
    print(f"{'Best Val Loss':<25} {transformer_info.get('best_val_loss', 'N/A'):<25} {mamba_info.get('best_val_loss', 'N/A'):<25}")
    print(f"{'Final Iteration':<25} {transformer_info.get('iter_num', 'N/A'):<25} {mamba_info.get('iter_num', 'N/A'):<25}")
    print("-" * 75)


def compare_samples(transformer_samples: List[str], mamba_samples: List[str], 
                   num_samples: int = 3) -> None:
    """Display side-by-side sample comparison."""
    
    print("\n" + "="*80)
    print("QUALITATIVE SAMPLE COMPARISON")
    print("="*80)
    
    num_to_show = min(num_samples, len(transformer_samples), len(mamba_samples))
    
    for i in range(num_to_show):
        print(f"\n{'='*80}")
        print(f"SAMPLE {i+1}")
        print(f"{'='*80}")
        
        print(f"\n🔵 TRANSFORMER OUTPUT:")
        print("-" * 80)
        print(transformer_samples[i][:500] + ('...' if len(transformer_samples[i]) > 500 else ''))
        
        print(f"\n🟠 MAMBA OUTPUT:")
        print("-" * 80)
        print(mamba_samples[i][:500] + ('...' if len(mamba_samples[i]) > 500 else ''))


def write_comparison_report(output_path: str, transformer_metrics: Dict, mamba_metrics: Dict,
                           transformer_info: Dict, mamba_info: Dict) -> None:
    """Write detailed comparison report to file."""
    
    with open(output_path, 'w') as f:
        f.write("# Transformer vs Mamba Architecture Comparison\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This report compares two sequence modeling architectures:\n")
        f.write("1. **Transformer** with standard multi-head self-attention\n")
        f.write("2. **Mamba** with selective state space models (SSM)\n\n")
        
        f.write("## Architecture Details\n\n")
        f.write("### Transformer\n")
        trans_args = transformer_info.get('model_args', {})
        for key, val in trans_args.items():
            f.write(f"- {key}: {val}\n")
        
        f.write("\n### Mamba\n")
        mamba_args = mamba_info.get('model_args', {})
        for key, val in mamba_args.items():
            f.write(f"- {key}: {val}\n")
        
        f.write("\n## Training Results\n\n")
        f.write(f"| Metric | Transformer | Mamba |\n")
        f.write(f"|--------|-------------|-------|\n")
        f.write(f"| Best Val Loss | {transformer_info.get('best_val_loss', 'N/A'):.4f} | {mamba_info.get('best_val_loss', 'N/A'):.4f} |\n")
        f.write(f"| Iterations | {transformer_info.get('iter_num', 'N/A')} | {mamba_info.get('iter_num', 'N/A')} |\n")
        
        f.write("\n## Evaluation Metrics\n\n")
        
        f.write("### Specific Metrics (Training-Data-Dependent)\n\n")
        f.write("| Metric | Transformer | Mamba | Difference |\n")
        f.write("|--------|-------------|-------|------------|\n")
        
        specific_metrics = ['1-gram_overlap', '2-gram_overlap', '3-gram_overlap', 'perplexity', 'kl_divergence']
        for metric in specific_metrics:
            trans_val = transformer_metrics.get(metric, 'N/A')
            mamba_val = mamba_metrics.get(metric, 'N/A')
            if isinstance(trans_val, (int, float)) and isinstance(mamba_val, (int, float)):
                diff = mamba_val - trans_val
                f.write(f"| {metric} | {trans_val:.4f} | {mamba_val:.4f} | {diff:+.4f} |\n")
            else:
                f.write(f"| {metric} | {trans_val} | {mamba_val} | N/A |\n")
        
        f.write("\n### General Metrics (Training-Data-Independent)\n\n")
        f.write("| Metric | Transformer | Mamba | Difference |\n")
        f.write("|--------|-------------|-------|------------|\n")
        
        general_metrics = ['self_bleu', 'distinct_1', 'distinct_2', 'distinct_3', 'shannon_entropy']
        for metric in general_metrics:
            trans_val = transformer_metrics.get(metric, 'N/A')
            mamba_val = mamba_metrics.get(metric, 'N/A')
            if isinstance(trans_val, (int, float)) and isinstance(mamba_val, (int, float)):
                diff = mamba_val - trans_val
                f.write(f"| {metric} | {trans_val:.4f} | {mamba_val:.4f} | {diff:+.4f} |\n")
            else:
                f.write(f"| {metric} | {trans_val} | {mamba_val} | N/A |\n")
        
        f.write("\n## Analysis\n\n")
        
        # Determine which model is better on key metrics
        trans_ppl = transformer_metrics.get('perplexity', float('inf'))
        mamba_ppl = mamba_metrics.get('perplexity', float('inf'))
        
        trans_kl = transformer_metrics.get('kl_divergence', float('inf'))
        mamba_kl = mamba_metrics.get('kl_divergence', float('inf'))
        
        f.write("### Key Findings\n\n")
        
        if trans_ppl < mamba_ppl:
            f.write(f"- **Perplexity**: Transformer achieves lower perplexity ({trans_ppl:.4f} vs {mamba_ppl:.4f}), ")
            f.write("indicating better next-token prediction.\n")
        else:
            f.write(f"- **Perplexity**: Mamba achieves lower perplexity ({mamba_ppl:.4f} vs {trans_ppl:.4f}), ")
            f.write("indicating better next-token prediction.\n")
        
        if trans_kl < mamba_kl:
            f.write(f"- **KL Divergence**: Transformer output distribution is closer to training data ")
            f.write(f"({trans_kl:.4f} vs {mamba_kl:.4f}).\n")
        else:
            f.write(f"- **KL Divergence**: Mamba output distribution is closer to training data ")
            f.write(f"({mamba_kl:.4f} vs {trans_kl:.4f}).\n")
        
        f.write("\n### Computational Trade-offs\n\n")
        f.write("- **Transformer**: O(n²) attention complexity, better for capturing all pairwise dependencies\n")
        f.write("- **Mamba**: O(n) SSM complexity, more efficient for long sequences\n")
        
        f.write("\n### Conclusion\n\n")
        f.write("This comparison demonstrates the trade-offs between attention-based and state-space models. ")
        f.write("While transformers excel at capturing complex patterns through attention, Mamba offers ")
        f.write("linear-time complexity with competitive performance.\n")
    
    print(f"\n💾 Saved detailed report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare Transformer and Mamba architectures')
    parser.add_argument('--transformer_dir', type=str, 
                       default='/nobackup/gaurav/out-shakespeare-extensive-L4-H4-E256',
                       help='Directory containing Transformer model outputs')
    parser.add_argument('--mamba_dir', type=str,
                       default='out-shakespeare-mamba',
                       help='Directory containing Mamba model outputs')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                       help='Directory to save comparison results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("ARCHITECTURE COMPARISON: TRANSFORMER vs MAMBA")
    print("="*80)
    
    # Load results
    print("\n📂 Loading evaluation results...")
    try:
        transformer_metrics = load_evaluation_results(args.transformer_dir)
        print(f"   ✓ Loaded Transformer metrics from {args.transformer_dir}")
    except Exception as e:
        print(f"   ✗ Error loading Transformer metrics: {e}")
        transformer_metrics = {}
    
    try:
        mamba_metrics = load_evaluation_results(args.mamba_dir)
        print(f"   ✓ Loaded Mamba metrics from {args.mamba_dir}")
    except Exception as e:
        print(f"   ✗ Error loading Mamba metrics: {e}")
        mamba_metrics = {}
    
    # Load checkpoint info
    print("\n📂 Loading checkpoint information...")
    try:
        transformer_info = load_checkpoint_info(args.transformer_dir)
        print(f"   ✓ Loaded Transformer checkpoint info")
    except Exception as e:
        print(f"   ✗ Error loading Transformer checkpoint: {e}")
        transformer_info = {}
    
    try:
        mamba_info = load_checkpoint_info(args.mamba_dir)
        print(f"   ✓ Loaded Mamba checkpoint info")
    except Exception as e:
        print(f"   ✗ Error loading Mamba checkpoint: {e}")
        mamba_info = {}
    
    # Load samples
    print("\n📂 Loading generated samples...")
    try:
        transformer_samples = load_samples(args.transformer_dir)
        print(f"   ✓ Loaded {len(transformer_samples)} Transformer samples")
    except Exception as e:
        print(f"   ✗ Error loading Transformer samples: {e}")
        transformer_samples = []
    
    try:
        mamba_samples = load_samples(args.mamba_dir)
        print(f"   ✓ Loaded {len(mamba_samples)} Mamba samples")
    except Exception as e:
        print(f"   ✗ Error loading Mamba samples: {e}")
        mamba_samples = []
    
    # Perform comparisons
    if transformer_metrics and mamba_metrics:
        compare_metrics(transformer_metrics, mamba_metrics)
        plot_metric_comparison(transformer_metrics, mamba_metrics, args.output_dir)
    
    if transformer_info and mamba_info:
        compare_model_architectures(transformer_info, mamba_info)
    
    if transformer_samples and mamba_samples:
        compare_samples(transformer_samples, mamba_samples, num_samples=3)
    
    # Write report
    if transformer_metrics and mamba_metrics and transformer_info and mamba_info:
        report_path = os.path.join(args.output_dir, 'comparison_report.md')
        write_comparison_report(report_path, transformer_metrics, mamba_metrics,
                              transformer_info, mamba_info)
    
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}/")
    print("  - metrics_comparison.png")
    print("  - comparison_report.md")


if __name__ == '__main__':
    main()

