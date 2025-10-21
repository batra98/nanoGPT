#!/bin/bash
# Download pre-trained Shakespeare checkpoint from Hugging Face

set -e

echo "======================================================================"
echo "Downloading Pre-trained Shakespeare Checkpoint"
echo "======================================================================"

# Create output directory
OUTPUT_DIR="out-shakespeare-pretrained"
mkdir -p "$OUTPUT_DIR"

# Hugging Face URL
CHECKPOINT_URL="https://huggingface.co/batra98/hyperparameter-tuning-shakespeare/resolve/main/out-shakespeare-extensive-L4-H4-E256/ckpt.pt"
OUTPUT_FILE="$OUTPUT_DIR/ckpt.pt"

# Check if already downloaded
if [ -f "$OUTPUT_FILE" ]; then
    echo "✓ Checkpoint already exists at $OUTPUT_FILE"
    echo "File size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    read -p "Re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download."
        exit 0
    fi
fi

# Download with wget
echo ""
echo "Downloading from Hugging Face..."
echo "URL: $CHECKPOINT_URL"
echo "Output: $OUTPUT_FILE"
echo ""

if command -v wget &> /dev/null; then
    wget -O "$OUTPUT_FILE" "$CHECKPOINT_URL" --show-progress
elif command -v curl &> /dev/null; then
    curl -L -o "$OUTPUT_FILE" "$CHECKPOINT_URL" --progress-bar
else
    echo "Error: Neither wget nor curl found. Please install one of them."
    exit 1
fi

# Verify download
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "======================================================================"
    echo "✓ Download complete!"
    echo "======================================================================"
    echo "Checkpoint saved to: $OUTPUT_FILE"
    echo "File size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    echo "======================================================================"
else
    echo "Error: Download failed"
    exit 1
fi

