#!/bin/bash
# Complete RLHF/DPO Pipeline
# Usage: bash run_rlhf_pipeline.sh

set -e  # Exit on error

echo "======================================================================"
echo "RLHF/DPO Pipeline for Shakespeare"
echo "======================================================================"
echo ""

# Configuration
DEVICE="cuda"  # Change to "cpu" if no GPU
BASE_CHECKPOINT="out-shakespeare/ckpt.pt"
NUM_PAIRS=1000
NUM_EPOCHS_REWARD=10
NUM_EPOCHS_DPO=5
WANDB_ENABLED=false  # Set to true to enable wandb logging
WANDB_PROJECT="rlhf-shakespeare"  # Wandb project name

echo "Configuration:"
echo "  Device: $DEVICE"
echo "  Base checkpoint: $BASE_CHECKPOINT"
echo "  Preference pairs: $NUM_PAIRS"
echo "  Wandb logging: $WANDB_ENABLED"
if [ "$WANDB_ENABLED" = true ]; then
    echo "  Wandb project: $WANDB_PROJECT"
fi
echo ""

# Check if base checkpoint exists
if [ ! -f "$BASE_CHECKPOINT" ]; then
    echo "ERROR: Base checkpoint not found at $BASE_CHECKPOINT"
    echo "Please train a base model first or update BASE_CHECKPOINT variable"
    exit 1
fi

echo "======================================================================"
echo "Step 1/5: Generate Preference Data"
echo "======================================================================"
python3 generate_preference_data.py \
    --checkpoint_dir=out-shakespeare \
    --num_pairs=$NUM_PAIRS \
    --prompt_length=64 \
    --max_new_tokens=128 \
    --output_dir=data/preferences \
    --device=$DEVICE

echo ""
echo "======================================================================"
echo "Step 2/5: Train Reward Model"
echo "======================================================================"
WANDB_ARGS=""
if [ "$WANDB_ENABLED" = true ]; then
    WANDB_ARGS="--wandb_log --wandb_project=$WANDB_PROJECT --wandb_run_name=reward-model"
fi

python3 train_reward_model.py \
    --gpt_checkpoint=$BASE_CHECKPOINT \
    --train_data=data/preferences/train.pkl \
    --val_data=data/preferences/val.pkl \
    --out_dir=out-reward-model \
    --batch_size=32 \
    --num_epochs=$NUM_EPOCHS_REWARD \
    --learning_rate=1e-4 \
    --device=$DEVICE \
    $WANDB_ARGS

echo ""
echo "======================================================================"
echo "Step 3/5: Test Reward Model"
echo "======================================================================"
WANDB_ARGS=""
if [ "$WANDB_ENABLED" = true ]; then
    WANDB_ARGS="--wandb_log --wandb_project=$WANDB_PROJECT --wandb_run_name=reward-test"
fi

python3 test_reward_model.py \
    --reward_checkpoint=out-reward-model/best_model.pt \
    --gpt_checkpoint=$BASE_CHECKPOINT \
    --num_samples=50 \
    --max_new_tokens=200 \
    --device=$DEVICE \
    --output_file=results/reward_model_test.txt \
    $WANDB_ARGS

echo ""
echo "======================================================================"
echo "Step 4/5: Train with DPO"
echo "======================================================================"
WANDB_ARGS=""
if [ "$WANDB_ENABLED" = true ]; then
    WANDB_ARGS="--wandb_log --wandb_project=$WANDB_PROJECT --wandb_run_name=dpo-training"
fi

python3 train_dpo.py \
    --ref_checkpoint=$BASE_CHECKPOINT \
    --train_data=data/preferences/train.pkl \
    --val_data=data/preferences/val.pkl \
    --out_dir=out-dpo \
    --batch_size=16 \
    --num_epochs=$NUM_EPOCHS_DPO \
    --learning_rate=1e-6 \
    --beta=0.1 \
    --device=$DEVICE \
    $WANDB_ARGS

echo ""
echo "======================================================================"
echo "Step 5/5: Evaluate DPO Model"
echo "======================================================================"
WANDB_ARGS=""
if [ "$WANDB_ENABLED" = true ]; then
    WANDB_ARGS="--wandb_log --wandb_project=$WANDB_PROJECT --wandb_run_name=dpo-evaluation"
fi

python3 evaluate_dpo.py \
    --base_checkpoint=$BASE_CHECKPOINT \
    --dpo_checkpoint=out-dpo/ckpt.pt \
    --reward_checkpoint=out-reward-model/best_model.pt \
    --num_samples=50 \
    --device=$DEVICE \
    --output_dir=results \
    $WANDB_ARGS

echo ""
echo "======================================================================"
echo "RLHF/DPO Pipeline Complete!"
echo "======================================================================"
echo ""
echo "Results saved to:"
echo "  - results/reward_model_test.txt"
echo "  - results/dpo_evaluation.txt"
echo "  - results/dpo_comparison.png"
echo ""
echo "Models saved to:"
echo "  - out-reward-model/best_model.pt"
echo "  - out-dpo/ckpt.pt"
echo ""

