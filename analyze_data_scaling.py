"""
Analyze data scaling experiment results.

Creates plots showing how model performance improves with more training data.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# Map dataset names to numeric sizes for plotting
DATASET_SIZES = {
    '100k': 100_000,
    '500k': 500_000,
    '1m': 1_000_000,
    '5m': 5_000_000,
    '10m': 10_000_000,
}


def load_results(results_file='data_scaling_results.json'):
    """Load experimental results from JSON file."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Sort by dataset size
    results = sorted(results, key=lambda x: DATASET_SIZES[x['dataset_name']])
    
    return results


def create_plots(results, output_dir='plots', force_recreate=False):
    """Create all analysis plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if plots already exist
    expected_plots = [
        '01_data_size_vs_val_loss.png',
        '02_data_size_vs_specific_metrics.png',
        '03_data_size_vs_general_metrics.png',
        '04_training_efficiency.png',
        '05_comprehensive_dashboard.png',
    ]
    
    all_exist = all(os.path.exists(os.path.join(output_dir, p)) for p in expected_plots)
    
    if all_exist and not force_recreate:
        print(f"\n✓ All plots already exist in {output_dir}/")
        response = input("Re-create plots? [y/N]: ").strip().lower()
        if response != 'y':
            print("Skipping plot creation. Existing plots will be used.")
            return
        else:
            print("Re-creating all plots...")
    
    # Extract data
    dataset_names = [r['dataset_name'] for r in results]
    dataset_sizes = [DATASET_SIZES[name] for name in dataset_names]
    dataset_labels = [name.upper() for name in dataset_names]
    
    val_losses = [r['val_loss'] for r in results]
    perplexities = [r.get('perplexity', 0) for r in results]
    kl_divs = [r.get('kl_divergence', 0) for r in results]
    
    ngram1 = [r.get('ngram_overlap_1', 0) for r in results]
    ngram2 = [r.get('ngram_overlap_2', 0) for r in results]
    ngram3 = [r.get('ngram_overlap_3', 0) for r in results]
    
    self_bleus = [r.get('self_bleu', 0) for r in results]
    distinct1 = [r.get('distinct_1', 0) for r in results]
    distinct2 = [r.get('distinct_2', 0) for r in results]
    distinct3 = [r.get('distinct_3', 0) for r in results]
    entropies = [r.get('entropy', 0) for r in results]
    
    training_times = [r.get('training_time_min', 0) for r in results]
    final_iters = [r.get('final_iteration', 0) for r in results]
    
    print(f"Creating plots in {output_dir}/")
    
    # ========================================================================
    # Plot 1: Data Size vs Validation Loss
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dataset_sizes, val_losses, 'o-', linewidth=2, markersize=10, color='#2E86AB')
    ax.set_xlabel('Training Data Size (characters)', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Model Performance vs Dataset Size\nArchitecture: L4-H4-E256', 
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(dataset_sizes)
    ax.set_xticklabels(dataset_labels, rotation=45)
    
    # Annotate points
    for x, y, label in zip(dataset_sizes, val_losses, dataset_labels):
        ax.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 10), 
                   textcoords='offset points', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_data_size_vs_val_loss.png', dpi=150)
    plt.close()
    print("  ✓ Created: 01_data_size_vs_val_loss.png")
    
    # ========================================================================
    # Plot 2: Data Size vs Specific Metrics (Multi-line)
    # ========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # N-gram overlaps
    ax1.plot(dataset_sizes, ngram1, 'o-', label='Unigram', linewidth=2, markersize=8, color='#2E86AB')
    ax1.plot(dataset_sizes, ngram2, 's-', label='Bigram', linewidth=2, markersize=8, color='#06A77D')
    ax1.plot(dataset_sizes, ngram3, '^-', label='Trigram', linewidth=2, markersize=8, color='#F18F01')
    ax1.set_xlabel('Training Data Size (characters)', fontsize=12)
    ax1.set_ylabel('N-gram Overlap (%)', fontsize=12)
    ax1.set_title('N-gram Coverage vs Dataset Size', fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_xticks(dataset_sizes)
    ax1.set_xticklabels(dataset_labels, rotation=45)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Perplexity and KL Divergence (dual y-axis)
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(dataset_sizes, perplexities, 'o-', label='Perplexity', 
                     linewidth=2, markersize=8, color='#A23B72')
    line2 = ax2_twin.plot(dataset_sizes, kl_divs, 's-', label='KL Divergence', 
                          linewidth=2, markersize=8, color='#F18F01')
    ax2.set_xlabel('Training Data Size (characters)', fontsize=12)
    ax2.set_ylabel('Perplexity', fontsize=12, color='#A23B72')
    ax2_twin.set_ylabel('KL Divergence', fontsize=12, color='#F18F01')
    ax2.set_title('Quality Metrics vs Dataset Size', fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_xticks(dataset_sizes)
    ax2.set_xticklabels(dataset_labels, rotation=45)
    ax2.tick_params(axis='y', labelcolor='#A23B72')
    ax2_twin.tick_params(axis='y', labelcolor='#F18F01')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_data_size_vs_specific_metrics.png', dpi=150)
    plt.close()
    print("  ✓ Created: 02_data_size_vs_specific_metrics.png")
    
    # ========================================================================
    # Plot 3: Data Size vs General Metrics
    # ========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Self-BLEU (lower is better - more diverse)
    ax1.plot(dataset_sizes, self_bleus, 'o-', linewidth=2, markersize=10, color='#A23B72')
    ax1.set_xlabel('Training Data Size (characters)', fontsize=12)
    ax1.set_ylabel('Self-BLEU Score', fontsize=12)
    ax1.set_title('Diversity (Self-BLEU) vs Dataset Size\n(Lower = More Diverse)', 
                 fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_xticks(dataset_sizes)
    ax1.set_xticklabels(dataset_labels, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Distinct-n metrics
    ax2.plot(dataset_sizes, distinct1, 'o-', label='Distinct-1', linewidth=2, markersize=8, color='#2E86AB')
    ax2.plot(dataset_sizes, distinct2, 's-', label='Distinct-2', linewidth=2, markersize=8, color='#06A77D')
    ax2.plot(dataset_sizes, distinct3, '^-', label='Distinct-3', linewidth=2, markersize=8, color='#F18F01')
    ax2.set_xlabel('Training Data Size (characters)', fontsize=12)
    ax2.set_ylabel('Distinct-n Score (%)', fontsize=12)
    ax2.set_title('Lexical Diversity vs Dataset Size\n(Higher = More Diverse)', 
                 fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_xticks(dataset_sizes)
    ax2.set_xticklabels(dataset_labels, rotation=45)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_data_size_vs_general_metrics.png', dpi=150)
    plt.close()
    print("  ✓ Created: 03_data_size_vs_general_metrics.png")
    
    # ========================================================================
    # Plot 4: Training Efficiency
    # ========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training time
    ax1.plot(dataset_sizes, training_times, 'o-', linewidth=2, markersize=10, color='#2E86AB')
    ax1.set_xlabel('Training Data Size (characters)', fontsize=12)
    ax1.set_ylabel('Training Time (minutes)', fontsize=12)
    ax1.set_title('Training Duration vs Dataset Size', fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_xticks(dataset_sizes)
    ax1.set_xticklabels(dataset_labels, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Annotate times
    for x, y, label in zip(dataset_sizes, training_times, dataset_labels):
        ax1.annotate(f'{y:.1f}m', xy=(x, y), xytext=(0, 10), 
                    textcoords='offset points', ha='center', fontsize=9)
    
    # Iterations to convergence
    ax2.plot(dataset_sizes, final_iters, 's-', linewidth=2, markersize=10, color='#06A77D')
    ax2.set_xlabel('Training Data Size (characters)', fontsize=12)
    ax2.set_ylabel('Iterations (Early Stopping)', fontsize=12)
    ax2.set_title('Convergence Speed vs Dataset Size', fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_xticks(dataset_sizes)
    ax2.set_xticklabels(dataset_labels, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Annotate iterations
    for x, y, label in zip(dataset_sizes, final_iters, dataset_labels):
        ax2.annotate(f'{y}', xy=(x, y), xytext=(0, 10), 
                    textcoords='offset points', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_training_efficiency.png', dpi=150)
    plt.close()
    print("  ✓ Created: 04_training_efficiency.png")
    
    # ========================================================================
    # Plot 5: Comprehensive Dashboard (2x3 grid)
    # ========================================================================
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Subplot 1: Val Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(dataset_sizes, val_losses, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Dataset Size', fontsize=10)
    ax1.set_ylabel('Validation Loss', fontsize=10)
    ax1.set_title('Validation Loss', fontsize=11, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_xticks(dataset_sizes)
    ax1.set_xticklabels(dataset_labels, rotation=45, fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Perplexity
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(dataset_sizes, perplexities, 'o-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Dataset Size', fontsize=10)
    ax2.set_ylabel('Perplexity', fontsize=10)
    ax2.set_title('Perplexity (Quality)', fontsize=11, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_xticks(dataset_sizes)
    ax2.set_xticklabels(dataset_labels, rotation=45, fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: N-gram Overlaps
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(dataset_sizes, ngram1, 'o-', label='1-gram', linewidth=2, markersize=6)
    ax3.plot(dataset_sizes, ngram2, 's-', label='2-gram', linewidth=2, markersize=6)
    ax3.plot(dataset_sizes, ngram3, '^-', label='3-gram', linewidth=2, markersize=6)
    ax3.set_xlabel('Dataset Size', fontsize=10)
    ax3.set_ylabel('N-gram Overlap (%)', fontsize=10)
    ax3.set_title('N-gram Coverage', fontsize=11, fontweight='bold')
    ax3.set_xscale('log')
    ax3.set_xticks(dataset_sizes)
    ax3.set_xticklabels(dataset_labels, rotation=45, fontsize=8)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Self-BLEU
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(dataset_sizes, self_bleus, 'o-', linewidth=2, markersize=8, color='#F18F01')
    ax4.set_xlabel('Dataset Size', fontsize=10)
    ax4.set_ylabel('Self-BLEU', fontsize=10)
    ax4.set_title('Self-BLEU (Diversity)', fontsize=11, fontweight='bold')
    ax4.set_xscale('log')
    ax4.set_xticks(dataset_sizes)
    ax4.set_xticklabels(dataset_labels, rotation=45, fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Subplot 5: Distinct-n
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(dataset_sizes, distinct1, 'o-', label='Distinct-1', linewidth=2, markersize=6)
    ax5.plot(dataset_sizes, distinct2, 's-', label='Distinct-2', linewidth=2, markersize=6)
    ax5.plot(dataset_sizes, distinct3, '^-', label='Distinct-3', linewidth=2, markersize=6)
    ax5.set_xlabel('Dataset Size', fontsize=10)
    ax5.set_ylabel('Distinct-n (%)', fontsize=10)
    ax5.set_title('Lexical Diversity', fontsize=11, fontweight='bold')
    ax5.set_xscale('log')
    ax5.set_xticks(dataset_sizes)
    ax5.set_xticklabels(dataset_labels, rotation=45, fontsize=8)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Subplot 6: Training Time
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(dataset_sizes, training_times, 'o-', linewidth=2, markersize=8, color='#06A77D')
    ax6.set_xlabel('Dataset Size', fontsize=10)
    ax6.set_ylabel('Training Time (min)', fontsize=10)
    ax6.set_title('Training Duration', fontsize=11, fontweight='bold')
    ax6.set_xscale('log')
    ax6.set_xticks(dataset_sizes)
    ax6.set_xticklabels(dataset_labels, rotation=45, fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    fig.suptitle('Linux Kernel Data Scaling Analysis\nArchitecture: L4-H4-E256', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(f'{output_dir}/05_comprehensive_dashboard.png', dpi=150)
    plt.close()
    print("  ✓ Created: 05_comprehensive_dashboard.png")
    
    print(f"\n✓ All plots saved to {output_dir}/")


def analyze_results(results):
    """Perform analysis and print insights."""
    print("\n" + "="*70)
    print("DATA SCALING ANALYSIS")
    print("="*70)
    
    dataset_names = [r['dataset_name'] for r in results]
    val_losses = [r['val_loss'] for r in results]
    perplexities = [r.get('perplexity', 0) for r in results]
    
    # Find improvement rates
    print("\nPerformance Improvements:")
    for i in range(1, len(results)):
        prev_loss = val_losses[i-1]
        curr_loss = val_losses[i]
        improvement = ((prev_loss - curr_loss) / prev_loss) * 100
        size_increase = DATASET_SIZES[dataset_names[i]] / DATASET_SIZES[dataset_names[i-1]]
        
        print(f"  {dataset_names[i-1]} → {dataset_names[i]} ({size_increase:.1f}x data):")
        print(f"    Val Loss: {prev_loss:.4f} → {curr_loss:.4f} ({improvement:+.2f}%)")
    
    # Find "elbow point" (diminishing returns)
    print("\nDiminishing Returns Analysis:")
    improvements = []
    for i in range(1, len(results)):
        prev_loss = val_losses[i-1]
        curr_loss = val_losses[i]
        improvement = prev_loss - curr_loss
        improvements.append(improvement)
    
    if improvements:
        max_improvement_idx = improvements.index(max(improvements))
        print(f"  Largest improvement: {dataset_names[max_improvement_idx]} → {dataset_names[max_improvement_idx+1]}")
        print(f"    Absolute improvement: {max(improvements):.4f}")
        
        # Find where improvement drops below threshold
        threshold = max(improvements) * 0.3  # 30% of max improvement
        for i, imp in enumerate(improvements):
            if imp < threshold:
                print(f"  Elbow point (diminishing returns): ~{dataset_names[i]}")
                print(f"    Recommendation: Use at least {dataset_names[i]} for reasonable performance")
                break
    
    # Best configuration
    best_idx = val_losses.index(min(val_losses))
    print(f"\nBest Performance:")
    print(f"  Dataset: {dataset_names[best_idx]}")
    print(f"  Val Loss: {val_losses[best_idx]:.4f}")
    print(f"  Perplexity: {perplexities[best_idx]:.4f}")
    
    print("\n" + "="*70)


def upload_to_wandb(results, output_dir='plots'):
    """Upload summary plots to WandB."""
    if not WANDB_AVAILABLE:
        print("WandB not available, skipping upload")
        return
    
    try:
        run = wandb.init(
            project='linux-kernel-data-scaling',
            name='data-scaling-summary',
            job_type='analysis'
        )
        
        # Upload all plots
        plot_files = [
            '01_data_size_vs_val_loss.png',
            '02_data_size_vs_specific_metrics.png',
            '03_data_size_vs_general_metrics.png',
            '04_training_efficiency.png',
            '05_comprehensive_dashboard.png',
        ]
        
        for plot_file in plot_files:
            plot_path = os.path.join(output_dir, plot_file)
            if os.path.exists(plot_path):
                wandb.log({f"analysis/{plot_file[:-4]}": wandb.Image(plot_path)})
        
        # Log summary table
        table = wandb.Table(
            columns=['dataset', 'size', 'val_loss', 'perplexity', 'kl_div', 'self_bleu', 'training_time'],
            data=[
                [
                    r['dataset_name'],
                    r['dataset_desc'],
                    r['val_loss'],
                    r.get('perplexity', 0),
                    r.get('kl_divergence', 0),
                    r.get('self_bleu', 0),
                    r.get('training_time_min', 0)
                ]
                for r in results
            ]
        )
        wandb.log({"analysis/summary_table": table})
        
        wandb.finish()
        print("✓ Uploaded results to WandB")
        
    except Exception as e:
        print(f"✗ WandB upload failed: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze data scaling experiment results')
    parser.add_argument('--force', action='store_true', help='Force re-creation of plots')
    args = parser.parse_args()
    
    print("="*70)
    print("ANALYZING DATA SCALING EXPERIMENT RESULTS")
    print("="*70)
    
    # Load results
    try:
        results = load_results('data_scaling_results.json')
        print(f"\n✓ Loaded {len(results)} experimental results")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("  Please run: python run_data_scaling_experiment.py --num_gpus 8")
        return
    
    # Create plots
    print("\nGenerating plots...")
    create_plots(results, output_dir='plots', force_recreate=args.force)
    
    # Analyze results
    analyze_results(results)
    
    # Upload to WandB
    print("\nUploading to WandB...")
    upload_to_wandb(results, output_dir='plots')
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nPlots saved in: plots/")
    print(f"Results saved in: data_scaling_results.json")


if __name__ == '__main__':
    main()

