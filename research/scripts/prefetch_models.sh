#!/usr/bin/env bash
set -euo pipefail

# Prefetch all benchmark and scoring models into the Hugging Face cache.
# Requires: huggingface-cli installed and HUGGINGFACE_HUB_TOKEN for gated models.

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found. Install with: pip install huggingface-hub"
  exit 1
fi

if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "Warning: HUGGINGFACE_HUB_TOKEN not set. Gated models may fail to download."
fi

MODELS=(
  "black-forest-labs/FLUX.1-dev"                       # flux_dev
  "SG161222/Realistic_Vision_V5.1_noVAE"               # realvis_xl
  "stabilityai/stable-diffusion-3-medium"              # sd3_medium
  "ByteDance/SDXL-Lightning"                           # sdxl_lightning
  "stabilityai/stable-diffusion-xl-base-1.0"           # sdxl
  "DeepFloyd/IF-I-XL-v1.0"                             # deepfloyd
  "HiDream-ai/HiDream-I1-Full"                         # hidream / logo_sdxl
  "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"           # CLIP scoring
  "shunk031/Aesthetic-Scorer"                          # aesthetic scoring
)

echo "Downloading models into HF cache..."
for model in "${MODELS[@]}"; do
  echo ">>> $model"
  huggingface-cli download "$model" \
    --cache-dir "${HF_HOME:-$HOME/.cache/huggingface}" \
    --resume-download \
    --local-dir-use-symlinks False || true
done

echo "Done. Cache path: ${HF_HOME:-$HOME/.cache/huggingface}"
