# RLHF/DPO Implementation Guide

Complete implementation of RLHF using Direct Preference Optimization (DPO) on Shakespeare text.

## Overview

This implements a full RLHF pipeline:
1. **Preference Heuristic**: Prefer text with more dialogue structure (character names + colons)
2. **Reward Model**: Train a classifier to predict preference
3. **DPO Training**: Align the model using Direct Preference Optimization
4. **Evaluation**: Compare base vs aligned model

## Prerequisites

### Prepare Data

Shakespeare BPE dataset is already prepared:
- `data/shakespeare/train.bin` (301,966 tokens)
- `data/shakespeare/val.bin` (36,059 tokens)

## Step-by-Step Execution

### Step 0: Train Base Model (if needed)

If you don't have a pretrained Shakespeare model, train one first:

```bash
# Quick training config
python3 train.py \
    --dataset=shakespeare \
    --out_dir=out-shakespeare \
    --batch_size=64 \
    --block_size=256 \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=256 \
    --max_iters=5000 \
    --eval_interval=500 \
    --learning_rate=1e-3 \
    --device=cuda \
    --compile=False
```

This takes ~10-15 minutes on a GPU.

### Step 1: Generate Preference Data

Generate pairs of completions and label them using dialogue density:

```bash
python3 generate_preference_data.py \
    --checkpoint_dir=out-shakespeare \
    --num_pairs=1000 \
    --prompt_length=64 \
    --max_new_tokens=128 \
    --output_dir=data/preferences \
    --device=cuda
```

**Output:**
- `data/preferences/train.pkl` (800 pairs)
- `data/preferences/val.pkl` (200 pairs)

**Expected time:** 5-10 minutes

### Step 2: Train Reward Model

Train a reward model to predict dialogue density preference:

```bash
python3 train_reward_model.py \
    --gpt_checkpoint=out-shakespeare/ckpt.pt \
    --train_data=data/preferences/train.pkl \
    --val_data=data/preferences/val.pkl \
    --out_dir=out-reward-model \
    --batch_size=32 \
    --num_epochs=10 \
    --learning_rate=1e-4 \
    --device=cuda
```

**Output:**
- `out-reward-model/best_model.pt`

**Expected time:** 2-3 minutes
**Expected accuracy:** 60-75%

### Step 3: Test Reward Model

Validate that the reward model learned the preference:

```bash
python3 test_reward_model.py \
    --reward_checkpoint=out-reward-model/best_model.pt \
    --gpt_checkpoint=out-shakespeare/ckpt.pt \
    --num_samples=50 \
    --max_new_tokens=200 \
    --device=cuda \
    --output_file=results/reward_model_test.txt
```

**Output:**
- `results/reward_model_test.txt` (high/low reward samples)

**What to check:**
- High-reward samples should have more "CHARACTER:" patterns
- Correlation between reward and actual dialogue density should be positive
- Expected correlation: 0.3-0.6

### Step 4: Train with DPO

Align the model using Direct Preference Optimization:

```bash
python3 train_dpo.py \
    --ref_checkpoint=out-shakespeare/ckpt.pt \
    --train_data=data/preferences/train.pkl \
    --val_data=data/preferences/val.pkl \
    --out_dir=out-dpo \
    --batch_size=16 \
    --num_epochs=5 \
    --learning_rate=1e-6 \
    --beta=0.1 \
    --device=cuda
```

**Output:**
- `out-dpo/ckpt.pt`

**Expected time:** 5-10 minutes
**Expected accuracy:** 55-70%

**Key parameters:**
- `beta=0.1`: Controls how much the model can deviate from reference (higher = more conservative)
- `learning_rate=1e-6`: Very small to avoid destroying pretrained knowledge

### Step 5: Evaluate DPO Model

Compare base model vs DPO-aligned model:

```bash
python3 evaluate_dpo.py \
    --base_checkpoint=out-shakespeare/ckpt.pt \
    --dpo_checkpoint=out-dpo/ckpt.pt \
    --reward_checkpoint=out-reward-model/best_model.pt \
    --num_samples=50 \
    --device=cuda \
    --output_dir=results
```

**Output:**
- `results/dpo_evaluation.txt` (detailed comparison)
- `results/dpo_comparison.png` (histograms)

**Expected improvements:**
- Reward score: +0.1 to +0.5
- Dialogue density: +0.3 to +1.0 markers per 100 chars
- KL divergence: 0.01 to 0.1 (measures deviation from base)

## Complete Pipeline (One Script)

For convenience, you can run everything in sequence:

```bash
#!/bin/bash

# Step 1: Generate preferences
python3 generate_preference_data.py \
    --checkpoint_dir=out-shakespeare \
    --num_pairs=1000 \
    --device=cuda

# Step 2: Train reward model
python3 train_reward_model.py \
    --gpt_checkpoint=out-shakespeare/ckpt.pt \
    --num_epochs=10 \
    --device=cuda

# Step 3: Test reward model
python3 test_reward_model.py \
    --device=cuda

# Step 4: Train with DPO
python3 train_dpo.py \
    --ref_checkpoint=out-shakespeare/ckpt.pt \
    --num_epochs=5 \
    --device=cuda

# Step 5: Evaluate
python3 evaluate_dpo.py \
    --device=cuda

echo "RLHF pipeline complete!"
```

## File Structure

```
nanoGPT/
├── preference_heuristic.py        # Dialogue density metric
├── generate_preference_data.py    # Create preference pairs
├── reward_model.py                # Reward model architecture
├── train_reward_model.py          # Train reward model
├── test_reward_model.py           # Test reward model
├── train_dpo.py                   # DPO training
├── evaluate_dpo.py                # Final evaluation
│
├── data/
│   ├── shakespeare/               # Shakespeare BPE dataset
│   │   ├── train.bin
│   │   └── val.bin
│   └── preferences/               # Generated preference data
│       ├── train.pkl
│       └── val.pkl
│
├── out-shakespeare/               # Base model checkpoint
│   └── ckpt.pt
├── out-reward-model/              # Reward model checkpoint
│   └── best_model.pt
├── out-dpo/                       # DPO-aligned model
│   └── ckpt.pt
│
└── results/                       # Evaluation results
    ├── reward_model_test.txt
    ├── dpo_evaluation.txt
    └── dpo_comparison.png
```

## Key Hyperparameters

### Preference Data Generation
- `num_pairs=1000`: Number of preference pairs (more = better reward model, but slower)
- `prompt_length=64`: Length of prompt for generation
- `max_new_tokens=128`: Length of completion

### Reward Model Training
- `num_epochs=10`: Usually converges in 5-15 epochs
- `learning_rate=1e-4`: Standard for fine-tuning MLP head
- `batch_size=32`: Adjust based on GPU memory

### DPO Training
- `beta=0.1`: Key parameter! Controls strength of alignment
  - Higher (0.5): Model stays closer to reference, more stable
  - Lower (0.01): Model can change more, potentially better alignment
- `learning_rate=1e-6`: Very small to preserve base model quality
- `num_epochs=3-5`: DPO usually needs few epochs

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `block_size` to 128
- Use `--device=cpu` (much slower)

### Reward Model Poor Accuracy (<55%)
- Generate more preference pairs (`--num_pairs=2000`)
- Check preference data is balanced (A vs B preferences)
- Try more epochs or different learning rate

### DPO Not Improving
- Check reward model works first (Step 3)
- Try different `beta` values (0.05, 0.2)
- Reduce learning rate (1e-7)
- Check KL divergence isn't too high (>0.5 = model drifted too far)

### Preference Data Imbalanced
If one completion is always preferred:
- Use different temperature ranges in generation
- Adjust preference heuristic threshold
- Check dialogue density distribution in base model

## Expected Total Time

- **With GPU**: 30-45 minutes total
- **With CPU**: 3-4 hours total

## Citation

If using this implementation, please cite:

```
DPO Paper: Rafailov et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)
nanoGPT: Andrej Karpathy's nanoGPT repository
```

## Notes

- The dialogue density heuristic is simple but effective for Shakespeare
- For other domains, modify `preference_heuristic.py`
- DPO is more stable than PPO-style RLHF
- Results may vary - run multiple times with different seeds for robustness

