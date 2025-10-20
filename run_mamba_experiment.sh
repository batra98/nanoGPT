#!/bin/bash
# Complete Mamba SSM Experiment Runner
# This script trains Mamba, evaluates both models, and generates comparison

set -e  # Exit on error

echo "=============================================="
echo "Mamba SSM vs Transformer Comparison"
echo "=============================================="
echo ""

# Configuration
MAMBA_OUT_DIR="out-shakespeare-mamba"
TRANSFORMER_OUT_DIR="/nobackup/gaurav/out-shakespeare-extensive-L4-H4-E256"
COMPARISON_DIR="comparison_results"

# Step 1: Train Mamba Model
echo "Step 1/4: Training Mamba model..."
echo "----------------------------------------------"
python3 train_mamba.py config/train_shakespeare_mamba.py

if [ ! -f "$MAMBA_OUT_DIR/ckpt.pt" ]; then
    echo "ERROR: Mamba training failed - checkpoint not found"
    exit 1
fi
echo "✓ Mamba training complete"
echo ""

# Step 2: Evaluate Mamba
echo "Step 2/4: Evaluating Mamba model..."
echo "----------------------------------------------"
python3 sample_and_evaluate_mamba.py \
    --out_dir "$MAMBA_OUT_DIR" \
    --num_samples 10 \
    --max_new_tokens 1000 \
    --device cuda

if [ ! -f "$MAMBA_OUT_DIR/evaluation_metrics.json" ]; then
    echo "ERROR: Mamba evaluation failed - metrics not found"
    exit 1
fi
echo "✓ Mamba evaluation complete"
echo ""

# Step 3: Evaluate Transformer (if needed)
echo "Step 3/4: Checking Transformer evaluation..."
echo "----------------------------------------------"
if [ ! -f "$TRANSFORMER_OUT_DIR/evaluation_metrics.json" ]; then
    echo "Transformer metrics not found. Evaluating..."
    python3 sample_and_evaluate.py \
        --out_dir "$TRANSFORMER_OUT_DIR" \
        --num_samples 10 \
        --max_new_tokens 1000 \
        --device cuda
    echo "✓ Transformer evaluation complete"
else
    echo "✓ Transformer metrics already exist"
fi
echo ""

# Step 4: Compare architectures
echo "Step 4/4: Comparing architectures..."
echo "----------------------------------------------"
python3 compare_architectures.py \
    --transformer_dir "$TRANSFORMER_OUT_DIR" \
    --mamba_dir "$MAMBA_OUT_DIR" \
    --output_dir "$COMPARISON_DIR"

echo ""
echo "=============================================="
echo "✅ EXPERIMENT COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - $COMPARISON_DIR/metrics_comparison.png"
echo "  - $COMPARISON_DIR/comparison_report.md"
echo ""
echo "Quick summary:"
python3 << EOF
import json

print("\nTransformer (L4-H4-E256):")
try:
    with open('$TRANSFORMER_OUT_DIR/evaluation_metrics.json') as f:
        trans = json.load(f)
    print(f"  Perplexity: {trans.get('perplexity', 'N/A'):.4f}")
    print(f"  KL Divergence: {trans.get('kl_divergence', 'N/A'):.4f}")
    print(f"  Self-BLEU: {trans.get('self_bleu', 'N/A'):.4f}")
except:
    print("  (metrics not available)")

print("\nMamba (L4-D256-S16):")
try:
    with open('$MAMBA_OUT_DIR/evaluation_metrics.json') as f:
        mamba = json.load(f)
    print(f"  Perplexity: {mamba.get('perplexity', 'N/A'):.4f}")
    print(f"  KL Divergence: {mamba.get('kl_divergence', 'N/A'):.4f}")
    print(f"  Self-BLEU: {mamba.get('self_bleu', 'N/A'):.4f}")
except:
    print("  (metrics not available)")
EOF

echo ""
echo "View full report: cat $COMPARISON_DIR/comparison_report.md"
echo "View plot: xdg-open $COMPARISON_DIR/metrics_comparison.png"
echo ""

