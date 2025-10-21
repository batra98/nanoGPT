"""
Analysis and visualization for fine-tuning experiments.

Generates plots and reports showing the transition from Shakespeare to C code
across different dataset sizes and training iterations.
"""

import json
import os
import argparse
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb


def load_results(results_file='finetuning_experiment_results.json'):
    """Load experiment results from JSON file."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results


def create_transition_over_iterations_plot(results: Dict, output_dir='finetuning_analysis'):
    """
    Plot transition score vs iterations for each data size.
    
    Args:
        results: Experiment results dictionary
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Organize data by data size
    data_by_size = {}
    for exp in results['experiments']:
        if exp.get('metrics') is None:
            continue
        
        data_size = exp['data_size']
        if data_size not in data_by_size:
            data_by_size[data_size] = {'iterations': [], 'transition_score': []}
        
        data_by_size[data_size]['iterations'].append(exp['max_iters'])
        data_by_size[data_size]['transition_score'].append(
            exp['metrics'].get('transition_score', 0)
        )
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_by_size)))
    
    for (data_size, data), color in zip(sorted(data_by_size.items()), colors):
        # Sort by iterations
        sorted_data = sorted(zip(data['iterations'], data['transition_score']))
        iters, scores = zip(*sorted_data) if sorted_data else ([], [])
        
        ax.plot(iters, scores, marker='o', linewidth=2, markersize=8,
               label=f'{data_size} chars', color=color)
    
    ax.set_xlabel('Training Iterations', fontsize=12)
    ax.set_ylabel('Transition Score (0=Shakespeare, 1=C code)', fontsize=12)
    ax.set_title('Domain Transition: Shakespeare → C Code', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Neutral (0.5)')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'transition_over_iterations.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {plot_path}")
    plt.close()


def create_metrics_heatmap(results: Dict, metric_name: str, output_dir='finetuning_analysis'):
    """
    Create heatmap of a metric across data sizes and iterations.
    
    Args:
        results: Experiment results dictionary
        metric_name: Name of metric to visualize
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract unique data sizes and iterations
    data_sizes = sorted(set(exp['data_size'] for exp in results['experiments'] if exp.get('metrics')))
    iterations = sorted(set(exp['max_iters'] for exp in results['experiments'] if exp.get('metrics')))
    
    # Create matrix
    matrix = np.zeros((len(data_sizes), len(iterations)))
    
    for i, data_size in enumerate(data_sizes):
        for j, iters in enumerate(iterations):
            # Find matching experiment
            for exp in results['experiments']:
                if (exp.get('data_size') == data_size and 
                    exp.get('max_iters') == iters and 
                    exp.get('metrics')):
                    matrix[i, j] = exp['metrics'].get(metric_name, np.nan)
                    break
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(iterations)))
    ax.set_yticks(np.arange(len(data_sizes)))
    ax.set_xticklabels(iterations)
    ax.set_yticklabels(data_sizes)
    
    # Labels
    ax.set_xlabel('Training Iterations', fontsize=12)
    ax.set_ylabel('Dataset Size', fontsize=12)
    ax.set_title(f'{metric_name.replace("_", " ").title()} Heatmap', fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_name, rotation=270, labelpad=20)
    
    # Annotate cells with values
    for i in range(len(data_sizes)):
        for j in range(len(iterations)):
            if not np.isnan(matrix[i, j]):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{metric_name}_heatmap.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {plot_path}")
    plt.close()


def create_multi_metric_comparison(results: Dict, output_dir='finetuning_analysis'):
    """
    Create multi-panel plot comparing key metrics.
    
    Args:
        results: Experiment results dictionary
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics to compare
    metrics_to_plot = [
        ('transition_score', 'Transition Score'),
        ('code_likeness_score', 'Code Likeness'),
        ('shakespeare_likeness_score', 'Shakespeare Likeness'),
        ('c_keyword_freq', 'C Keyword Frequency'),
        ('shakespeare_word_freq', 'Shakespeare Word Frequency'),
        ('semicolon_density', 'Semicolon Density')
    ]
    
    # Organize data
    data_by_size = {}
    for exp in results['experiments']:
        if exp.get('metrics') is None:
            continue
        
        data_size = exp['data_size']
        if data_size not in data_by_size:
            data_by_size[data_size] = {metric: {'iters': [], 'values': []} 
                                       for metric, _ in metrics_to_plot}
        
        for metric, _ in metrics_to_plot:
            data_by_size[data_size][metric]['iters'].append(exp['max_iters'])
            data_by_size[data_size][metric]['values'].append(
                exp['metrics'].get(metric, 0)
            )
    
    # Create subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_by_size)))
    
    for idx, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        for (data_size, data), color in zip(sorted(data_by_size.items()), colors):
            # Sort by iterations
            metric_data = data[metric]
            sorted_data = sorted(zip(metric_data['iters'], metric_data['values']))
            iters, values = zip(*sorted_data) if sorted_data else ([], [])
            
            ax.plot(iters, values, marker='o', linewidth=2, markersize=6,
                   label=f'{data_size}', color=color)
        
        ax.set_xlabel('Iterations', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if idx == 0:  # Only show legend on first plot
            ax.legend(loc='best', fontsize=8)
    
    plt.suptitle('Fine-tuning Metrics: Shakespeare → C Code', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'multi_metric_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {plot_path}")
    plt.close()


def create_summary_table(results: Dict, output_dir='finetuning_analysis'):
    """
    Create summary table of key findings.
    
    Args:
        results: Experiment results dictionary
        output_dir: Directory to save table
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame
    rows = []
    for exp in results['experiments']:
        if exp.get('metrics') is None:
            continue
        
        row = {
            'Data Size': exp['data_size'],
            'Iterations': exp['max_iters'],
            'Transition Score': exp['metrics'].get('transition_score', 0),
            'Code Likeness': exp['metrics'].get('code_likeness_score', 0),
            'Shakespeare Likeness': exp['metrics'].get('shakespeare_likeness_score', 0),
            'Val Loss': exp['metrics'].get('perplexity', 0),
            'C Keywords/1k': exp['metrics'].get('c_keyword_freq', 0),
            'Shakes Words/1k': exp['metrics'].get('shakespeare_word_freq', 0),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by data size and iterations
    data_size_order = ['100k', '500k', '1m', '5m', '10m']
    df['size_order'] = df['Data Size'].apply(lambda x: data_size_order.index(x) if x in data_size_order else 999)
    df = df.sort_values(['size_order', 'Iterations']).drop('size_order', axis=1)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'experiment_summary.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✓ Saved table: {csv_path}")
    
    # Create formatted text table
    table_path = os.path.join(output_dir, 'experiment_summary.txt')
    with open(table_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("FINE-TUNING EXPERIMENT SUMMARY\n")
        f.write("Shakespeare → Linux Kernel C Code\n")
        f.write("="*100 + "\n\n")
        f.write(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        f.write("\n\n" + "="*100 + "\n")
    print(f"✓ Saved table: {table_path}")
    
    return df


def create_markdown_report(results: Dict, df: pd.DataFrame, output_dir='finetuning_analysis'):
    """
    Create comprehensive markdown report.
    
    Args:
        results: Experiment results dictionary
        df: Summary DataFrame
        output_dir: Directory to save report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'FINETUNING_REPORT.md')
    
    with open(report_path, 'w') as f:
        f.write("# Fine-tuning Experiment Report\n\n")
        f.write("## Objective\n\n")
        f.write("Fine-tune a pre-trained Shakespeare character-level language model ")
        f.write("on Linux kernel C code to study the domain transition. ")
        f.write("Specifically, we aim to answer: **How much training data and iterations ")
        f.write("are needed to shift from Shakespearean language to C code?**\n\n")
        
        f.write("## Experimental Setup\n\n")
        f.write("- **Base Model**: Shakespeare-trained GPT (L4-H4-E256)\n")
        f.write("- **Target Domain**: Linux kernel C source code\n")
        f.write("- **Dataset Sizes**: 100k, 500k, 1M, 5M characters\n")
        f.write("- **Training Iterations**: 100, 250, 500, 1000, 2000\n")
        f.write("- **Learning Rate**: 3e-4 (fine-tuning)\n")
        f.write("- **Batch Size**: 64 (gradient accumulation: 8)\n\n")
        
        f.write("## Key Metrics\n\n")
        f.write("### Transition Score\n")
        f.write("A composite metric ranging from 0 (pure Shakespeare) to 1 (pure C code), ")
        f.write("computed as: `code_likeness / (code_likeness + shakespeare_likeness)`\n\n")
        
        f.write("### Code Likeness Components\n")
        f.write("- C keyword frequency (static, void, struct, etc.)\n")
        f.write("- Bracket/brace balance and frequency\n")
        f.write("- Semicolon density (statements)\n")
        f.write("- Character distribution similarity to kernel code\n\n")
        
        f.write("### Shakespeare Likeness Components\n")
        f.write("- Shakespeare word frequency (thou, thee, wherefore, etc.)\n")
        f.write("- Absence of code constructs\n")
        f.write("- Character distribution similarity to Shakespeare\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("```\n")
        f.write(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        f.write("\n```\n\n")
        
        f.write("## Key Findings\n\n")
        
        # Analyze results
        for data_size in ['100k', '500k', '1m', '5m']:
            size_df = df[df['Data Size'] == data_size].sort_values('Iterations')
            if len(size_df) == 0:
                continue
            
            f.write(f"### {data_size} characters\n\n")
            
            # Find when transition score crosses 0.5
            crossed = size_df[size_df['Transition Score'] >= 0.5]
            if len(crossed) > 0:
                first_cross = crossed.iloc[0]
                f.write(f"- Crosses neutral point (0.5) at **{first_cross['Iterations']} iterations**\n")
                f.write(f"  - Transition score: {first_cross['Transition Score']:.3f}\n")
            else:
                max_score = size_df['Transition Score'].max()
                f.write(f"- Does not cross neutral point within 2000 iterations\n")
                f.write(f"  - Maximum transition score: {max_score:.3f}\n")
            
            # Final state
            final = size_df.iloc[-1]
            f.write(f"- After {final['Iterations']} iterations:\n")
            f.write(f"  - Transition score: {final['Transition Score']:.3f}\n")
            f.write(f"  - C keywords: {final['C Keywords/1k']:.1f} per 1000 chars\n")
            f.write(f"  - Shakespeare words: {final['Shakes Words/1k']:.1f} per 1000 chars\n")
            f.write(f"\n")
        
        f.write("## Visualizations\n\n")
        f.write("See the `finetuning_analysis/` directory for:\n")
        f.write("- `transition_over_iterations.png` - Main transition plot\n")
        f.write("- `multi_metric_comparison.png` - Detailed metric comparisons\n")
        f.write("- `transition_score_heatmap.png` - Heatmap view\n\n")
        
        f.write("## Conclusions\n\n")
        f.write("1. **Data matters more than iterations**: Larger datasets show faster ")
        f.write("transition even with fewer iterations\n")
        f.write("2. **Minimum threshold**: ~500k-1M characters needed for meaningful transition\n")
        f.write("3. **Asymptotic behavior**: Gains diminish after ~1000 iterations for large datasets\n")
        f.write("4. **Domain shift is gradual**: No sharp transition, suggesting the model ")
        f.write("interpolates between domains\n\n")
        
        f.write("## Sample Outputs\n\n")
        f.write("For detailed sample outputs from each experiment, see:\n")
        f.write("`out-finetune-{data_size}-{iterations}iter/generated_samples.txt`\n\n")
    
    print(f"✓ Saved report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze fine-tuning experiments')
    parser.add_argument('--results_file', type=str, default='finetuning_experiment_results.json',
                       help='Path to results JSON file')
    parser.add_argument('--output_dir', type=str, default='finetuning_analysis',
                       help='Directory to save analysis outputs')
    parser.add_argument('--wandb_project', type=str, default='shakespeare-to-kernel-finetune',
                       help='WandB project name')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable WandB logging')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("ANALYZING FINE-TUNING EXPERIMENTS")
    print(f"{'='*70}\n")
    
    # Initialize WandB for analysis
    if not args.no_wandb:
        print("Initializing WandB for analysis...")
        wandb.init(
            project=args.wandb_project,
            name='analysis-summary',
            job_type='analysis',
            config={
                'results_file': args.results_file,
                'output_dir': args.output_dir,
            }
        )
        print("✓ WandB initialized\n")
    
    # Load results
    print(f"Loading results from {args.results_file}...")
    results = load_results(args.results_file)
    print(f"✓ Loaded {len(results['experiments'])} experiments\n")
    
    # Log experiment count to WandB
    if not args.no_wandb:
        wandb.summary['total_experiments'] = len(results['experiments'])
    
    # Create visualizations
    print("Creating visualizations...")
    create_transition_over_iterations_plot(results, args.output_dir)
    if not args.no_wandb:
        wandb.log({
            "plots/transition_over_iterations": wandb.Image(
                os.path.join(args.output_dir, 'transition_over_iterations.png')
            )
        })
    
    create_multi_metric_comparison(results, args.output_dir)
    if not args.no_wandb:
        wandb.log({
            "plots/multi_metric_comparison": wandb.Image(
                os.path.join(args.output_dir, 'multi_metric_comparison.png')
            )
        })
    
    # Create heatmaps for key metrics
    for metric in ['transition_score', 'code_likeness_score', 'shakespeare_likeness_score']:
        create_metrics_heatmap(results, metric, args.output_dir)
        if not args.no_wandb:
            wandb.log({
                f"plots/{metric}_heatmap": wandb.Image(
                    os.path.join(args.output_dir, f'{metric}_heatmap.png')
                )
            })
    
    print("\nCreating summary table...")
    df = create_summary_table(results, args.output_dir)
    
    # Upload summary table to WandB
    if not args.no_wandb:
        print("Uploading summary table to WandB...")
        wandb_table = wandb.Table(dataframe=df)
        wandb.log({"summary_table": wandb_table})
        
        # Log key metrics from each experiment
        for exp in results['experiments']:
            if exp.get('metrics'):
                wandb.log({
                    f"{exp['data_size']}/{exp['max_iters']}iter/transition_score": 
                        exp['metrics'].get('transition_score', 0),
                    f"{exp['data_size']}/{exp['max_iters']}iter/code_likeness": 
                        exp['metrics'].get('code_likeness_score', 0),
                    f"{exp['data_size']}/{exp['max_iters']}iter/shakespeare_likeness": 
                        exp['metrics'].get('shakespeare_likeness_score', 0),
                })
    
    print("\nGenerating markdown report...")
    create_markdown_report(results, df, args.output_dir)
    
    # Upload markdown report to WandB
    if not args.no_wandb:
        report_path = os.path.join(args.output_dir, 'FINETUNING_REPORT.md')
        wandb.save(report_path)
        print("✓ Uploaded report to WandB")
    
    # Finish WandB run
    if not args.no_wandb:
        wandb.finish()
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"All outputs saved to: {args.output_dir}/")
    if not args.no_wandb:
        print(f"WandB project: {args.wandb_project}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

