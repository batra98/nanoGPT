# Fine-tuning Shakespeare → Kernel Code

This directory contains scripts for fine-tuning a pre-trained Shakespeare model on Linux kernel C code to study domain transition.

## Objective

Answer the question: **How much training data and iterations are needed to shift from Shakespearean language to C code output?**

## Prerequisites

1. **Pre-trained Shakespeare Model**
   - Download from: https://huggingface.co/batra98/hyperparameter-tuning-shakespeare/tree/main/out-shakespeare-extensive-L4-H4-E256
   - Run: `bash download_shakespeare_checkpoint.sh`
   - Or manually place `ckpt.pt` in `out-shakespeare-pretrained/`

2. **Linux Kernel Datasets**
   - Should be prepared at: `/nobackup/gaurav/kernel_code/`
   - Required sizes: 100k, 500k, 1m, 5m
   - If not prepared, run: `python data/linux_kernel/prepare.py`

## Quick Start

### 1. Download Pre-trained Checkpoint

```bash
bash download_shakespeare_checkpoint.sh
```

Or manually:
```bash
mkdir -p out-shakespeare-pretrained
wget https://huggingface.co/batra98/hyperparameter-tuning-shakespeare/resolve/main/out-shakespeare-extensive-L4-H4-E256/ckpt.pt \
     -O out-shakespeare-pretrained/ckpt.pt
```

### 2. Run Experiments

Full experiment suite (all data sizes, all iteration counts):
```bash
python run_finetuning_experiment.py --num_gpus 8
```

Custom experiment:
```bash
python run_finetuning_experiment.py \
    --num_gpus 8 \
    --data_sizes 100k 500k 1m \
    --iterations 100 500 1000
```

### 3. Analyze Results

After experiments complete:
```bash
python analyze_finetuning.py
```

This generates:
- `finetuning_analysis/transition_over_iterations.png` - Main transition plot
- `finetuning_analysis/multi_metric_comparison.png` - Detailed metrics
- `finetuning_analysis/experiment_summary.csv` - Summary table
- `finetuning_analysis/FINETUNING_REPORT.md` - Full report

## File Structure

```
.
├── config/
│   └── finetune_kernel.py          # Fine-tuning configuration
├── domain_transition_metrics.py     # Transition metrics (Shakespeare ↔ C code)
├── run_finetuning_experiment.py     # Orchestration script
├── analyze_finetuning.py            # Analysis and visualization
├── download_shakespeare_checkpoint.sh  # Download helper script
├── out-shakespeare-pretrained/      # Pre-trained Shakespeare checkpoint
│   └── ckpt.pt
├── out-finetune-{size}-{iters}iter/ # Fine-tuned model outputs
│   ├── ckpt.pt                      # Fine-tuned checkpoint
│   ├── generated_samples.txt        # Generated samples
│   └── evaluation_metrics.json      # All metrics
├── finetuning_experiment_results.json  # Aggregated results
└── finetuning_analysis/             # Visualizations and reports
    ├── transition_over_iterations.png
    ├── multi_metric_comparison.png
    ├── experiment_summary.csv
    └── FINETUNING_REPORT.md
```

## Experimental Parameters

### Default Configuration

- **Data Sizes**: 100k, 500k, 1M, 5M characters
- **Iteration Counts**: 100, 250, 500, 1000, 2000
- **Learning Rate**: 3e-4 (fine-tuning optimized)
- **Batch Size**: 64 per GPU
- **Gradient Accumulation**: 8 steps
- **Total Experiments**: 4 sizes × 5 iteration counts = 20 experiments

### Metrics Tracked

**Standard Metrics** (from `evaluation_metrics.py`):
- N-gram overlap (1, 2, 3-grams)
- Perplexity
- KL divergence
- Self-BLEU (diversity)
- Distinct-n (uniqueness)
- Entropy

**Transition Metrics** (from `domain_transition_metrics.py`):
- **Transition Score**: 0 (Shakespeare) → 1 (C code)
- **Code Likeness Score**: How C-code-like the output is
- **Shakespeare Likeness Score**: How Shakespeare-like the output is
- **C Keyword Frequency**: `static`, `void`, `struct`, etc. per 1k chars
- **Shakespeare Word Frequency**: `thou`, `thee`, `wherefore`, etc. per 1k chars
- **Semicolon Density**: Statements per 1k chars
- **Bracket Balance**: How well-balanced `{}`, `[]`, `()` are
- **Character Distribution KL Divergence**: vs. kernel and Shakespeare references

## Usage Examples

### Run Single Experiment

Fine-tune on 500k dataset for 1000 iterations:
```bash
python -c "
from run_finetuning_experiment import finetune_model, evaluate_finetuned_model
out_dir = finetune_model('out-shakespeare-pretrained', '/nobackup/gaurav/kernel_code/500k', 1000, num_gpus=8)
evaluate_finetuned_model(out_dir)
"
```

### Resume Experiments

The script automatically resumes from where it left off:
```bash
# If interrupted, just run again
python run_finetuning_experiment.py --num_gpus 8
```

Results are stored in `finetuning_experiment_results.json` and updated after each experiment.

### Generate Samples Only

```bash
python sample.py \
    --out_dir=out-finetune-1m-1000iter \
    --num_samples=10 \
    --max_new_tokens=500
```

## Expected Runtime

Per experiment (on 8× RTX 2080 Ti):
- 100 iters: ~5 minutes
- 500 iters: ~20 minutes
- 1000 iters: ~40 minutes
- 2000 iters: ~80 minutes

**Total for all 20 experiments**: ~12-15 hours

## Monitoring Progress

Results are logged to WandB project: `shakespeare-to-kernel-finetune`

Each run is named: `finetune-{data_size}-{iterations}iter`

Monitor:
- Training/validation loss curves
- Transition metrics in real-time
- Generated samples

## Troubleshooting

### Checkpoint Not Found

```
FileNotFoundError: Checkpoint not found at out-shakespeare-pretrained/ckpt.pt
```

**Solution**: Run `bash download_shakespeare_checkpoint.sh`

### Dataset Not Found

```
⚠ Skipping {size}: dataset not found at /nobackup/gaurav/kernel_code/{size}
```

**Solution**: Run `python data/linux_kernel/prepare.py --output_dir /nobackup/gaurav/kernel_code`

### CUDA Out of Memory

**Solution**: Reduce batch size in `config/finetune_kernel.py`:
```python
batch_size = 32  # instead of 64
```

### Training Hangs

**Solution**: Check GPU utilization with `nvidia-smi`. If using DDP, ensure `gradient_accumulation_steps` is divisible by number of GPUs.

## Citation

If you use this code, please cite:
```
CS839 Assignment - Fine-tuning Shakespeare to Kernel Code
University of Wisconsin-Madison
2025
```

## References

- Base nanoGPT: https://github.com/karpathy/nanoGPT
- Shakespeare Model: https://huggingface.co/batra98/hyperparameter-tuning-shakespeare
- Linux Kernel: https://github.com/torvalds/linux

