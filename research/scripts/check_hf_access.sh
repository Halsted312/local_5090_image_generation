#!/usr/bin/env bash
set -euo pipefail

# Lightweight access check for benchmark models.
# Does not print tokens. Exits non-zero on failure.

MODELS=(
  "black-forest-labs/FLUX.1-dev"
  "SG161222/Realistic_Vision_V5.1_noVAE"
  "stabilityai/stable-diffusion-3-medium"
  "ByteDance/SDXL-Lightning"
  "stabilityai/stable-diffusion-xl-base-1.0"
  "DeepFloyd/IF-I-XL-v1.0"
  "HiDream-ai/HiDream-I1-Full"
  "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
  "shunk031/Aesthetic-Scorer"
)

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found. Install with: pip install huggingface-hub"
  exit 1
fi

echo "Checking HF auth (token must be set in env)..."
huggingface-cli whoami > /dev/null
echo "Auth OK."

for model in "${MODELS[@]}"; do
  echo -n "Checking $model ... "
  if huggingface-cli repo info "$model" > /dev/null; then
    echo "OK"
  else
    echo "FAILED (accept license? gated?)"
    exit 1
  fi
done

echo "All checks passed."
