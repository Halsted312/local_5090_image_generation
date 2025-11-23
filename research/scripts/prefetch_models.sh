#!/usr/bin/env bash
set -euo pipefail

# Prefetch all benchmark and scoring models into the Hugging Face cache.
# Requires: huggingface-cli installed and HUGGINGFACE_HUB_TOKEN for gated models.

HF_CLI="python -m huggingface_hub.cli.hf"

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
  "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE" # aesthetic scoring
)

echo "Downloading models into HF cache..."
python - <<'PY'
import os
from huggingface_hub import snapshot_download

models = [
    "black-forest-labs/FLUX.1-dev",
    "SG161222/Realistic_Vision_V5.1_noVAE",
    "stabilityai/stable-diffusion-3-medium",
    "ByteDance/SDXL-Lightning",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "DeepFloyd/IF-I-XL-v1.0",
    "HiDream-ai/HiDream-I1-Full",
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE",
]

cache_dir = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

for model in models:
    print(f">>> {model}")
    try:
        snapshot_download(
            repo_id=model,
            cache_dir=cache_dir,
            token=token,
            resume_download=True,
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        print(f"Failed: {model}: {exc}")

print(f"Done. Cache path: {cache_dir}")
PY
