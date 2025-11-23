# GRPO (RLVR) Implementation Guide

Complete implementation of GRPO (Group Relative Policy Optimization) with RL from Verifier Rewards.

## Overview

This implements GRPO training using a simple verifier function:
- **Verifier:** Count of 's' characters (case-insensitive), capped at 50
- **Objective:** Maximize verifier reward using importance sampling with KL penalty

## Prerequisites

- Base model trained: `out-shakespeare/ckpt.pt`
- Shakespeare BPE dataset prepared: `data/shakespeare/val.bin`

## Step-by-Step Execution

### Step 1: Prepare Prompts and Evaluate Baseline

Extract prompts and generate baseline completions:

```bash
python3 prepare_grpo_data.py \
    --checkpoint_dir=out-shakespeare \
    --num_prompts=100 \
    --prompt_length=64 \
    --max_new_tokens=200 \
    --output_dir=data/grpo \
    --device=cuda \
    --seed=42
```

**Output:**
- `data/grpo/prompts.pkl` (100 prompts)
- `data/grpo/baseline_results.pkl` (baseline scores and examples)

**Expected time:** 5-10 minutes

### Step 2: Train with GRPO

Train the model using GRPO:

```bash
# Single GPU
python3 train_grpo.py \
    --ref_checkpoint=out-shakespeare/ckpt.pt \
    --prompts_path=data/grpo/prompts.pkl \
    --out_dir=out-grpo \
    --batch_size=4 \
    --num_steps=1000 \
    --num_samples_per_prompt=4 \
    --max_new_tokens=200 \
    --learning_rate=1e-6 \
    --beta=0.1 \
    --device=cuda \
    --wandb_log \
    --wandb_project=grpo-rlvr \
    --wandb_run_name=grpo-training

# Multi-GPU (8 GPUs)
torchrun --standalone --nproc_per_node=8 train_grpo.py \
    --ref_checkpoint=out-shakespeare/ckpt.pt \
    --prompts_path=data/grpo/prompts.pkl \
    --out_dir=out-grpo \
    --batch_size=4 \
    --num_steps=1000 \
    --num_samples_per_prompt=4 \
    --max_new_tokens=200 \
    --learning_rate=1e-6 \
    --beta=0.1 \
    --device=cuda \
    --wandb_log \
    --wandb_project=grpo-rlvr \
    --wandb_run_name=grpo-training
```

**Output:**
- `out-grpo/ckpt.pt` (trained GRPO model)
- `out-grpo/reward_history.pkl` (training reward history)

**Expected time:** 10-20 minutes (depends on num_steps)

### Step 3: Evaluate GRPO Model

Compare base vs GRPO model:

```bash
python3 evaluate_grpo.py \
    --base_checkpoint=out-shakespeare/ckpt.pt \
    --grpo_checkpoint=out-grpo/ckpt.pt \
    --prompts_path=data/grpo/prompts.pkl \
    --num_samples=100 \
    --max_new_tokens=200 \
    --device=cuda \
    --output_dir=results \
    --wandb_log \
    --wandb_project=grpo-rlvr \
    --wandb_run_name=grpo-evaluation
```

**Output:**
- `results/grpo_comparison.png` (comparison plots)
- `results/grpo_evaluation.pkl` (evaluation results)

## Verifier Function

The verifier function is defined in `verifier.py`:

```python
v(y) = min(count('s' in y), Rmax)
```

- Case-insensitive counting
- Rmax = 50 (cap to prevent extreme values)
- Constraints: Single EOS token, max 200 tokens

## GRPO Objective

GRPO uses importance sampling with KL penalty:

```
L = -E[r(y) * w(y)] + β * KL(π_θ || π_ref)
```

where:
- `r(y) = v(y)` is the verifier reward
- `w(y) = π_θ(y|x) / π_ref(y|x)` is the importance weight
- `β = 0.1` is the KL penalty weight

## Hyperparameters

- **Beta (KL weight):** 0.1 (controls how close to stay to reference)
- **Learning rate:** 1e-6
- **Batch size:** 4 prompts per step
- **Samples per prompt:** 4
- **Training steps:** 1000-2000
- **Max new tokens:** 200

## Expected Results

- Baseline mean verifier score: ~15-25 (depends on text length)
- GRPO should improve mean score by 10-30%
- Training curve should show increasing mean reward over steps

