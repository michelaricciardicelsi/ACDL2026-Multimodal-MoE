#!/usr/bin/env bash
# run.sh — Full pipeline: install, train, evaluate, generate plots.
# Usage: bash run.sh
# Designed for Google Colab (T4/L4/A100), Linux, or macOS with CUDA.

set -euo pipefail

echo "========================================"
echo " Multimodal Sparse MoE — ACDL 2026 #15 "
echo "========================================"

# 1. Install dependencies
echo ""
echo "[1/4] Installing requirements..."
pip install -r requirements.txt --quiet

# 2. Smoke-test the MoE layer (CPU, no model download)
echo ""
echo "[2/4] Running MoE layer smoke test..."
python model/modeling_moe.py

# 3. Train (text-only path; encoders enabled but no image/audio data passed by default)
echo ""
echo "[3/4] Starting training..."
accelerate launch training/train.py --config training/config.yaml

# 4. Evaluate and generate plots
echo ""
echo "[4/4] Evaluating and generating plots..."
python evaluation/evaluate.py --config training/config.yaml --checkpoint checkpoints/epoch_1
python evaluation/metrics.py

echo ""
echo "Done. Checkpoints in ./checkpoints/, plots in ./output/"
