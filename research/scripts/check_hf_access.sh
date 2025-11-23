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

echo "Checking HF auth (token must be set in env)..."
python - <<'PY'
import os, sys
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

api = HfApi()
token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if not token:
    print("HUGGINGFACE_HUB_TOKEN not set; gated repos may fail.", file=sys.stderr)

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

try:
    info = api.whoami(token=token)
    print(f"Auth OK for user/organization: {info.get('name')}")
except Exception as exc:
    print(f"Auth check failed: {exc}", file=sys.stderr)
    sys.exit(1)

for m in models:
    try:
        api.repo_info(m, token=token)
        print(f"{m}: OK")
    except HfHubHTTPError as exc:
        print(f"{m}: FAILED ({exc})", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"{m}: FAILED ({exc})", file=sys.stderr)
        sys.exit(1)

print("All checks passed.")
PY
