# RTX 5090 Benchmark Setup Guide

## 🚨 Critical Issues Fixed

1. **Model Name Error**: `SG161222/Realistic_Vision_V5.1_noVAE` doesn't exist
   - ✅ Fixed to: `SG161222/RealVisXL_V4.0` (correct model name)

2. **Gated Models**: Several models require license acceptance on HuggingFace
   - FLUX.1-dev (black-forest-labs)
   - Stable Diffusion 3 Medium (stabilityai) 
   - DeepFloyd IF (DeepFloyd)

3. **RTX 5090 Support**: PyTorch needs sm_120 architecture support
   - ✅ Fixed: Using PyTorch 2.7.0 with CUDA 12.8

## 📋 Prerequisites

### 1. Accept Model Licenses (REQUIRED for gated models)

You must accept licenses for these gated models BEFORE downloading:

1. **FLUX.1-dev**: https://huggingface.co/black-forest-labs/FLUX.1-dev
   - Click "Agree and access repository"
   
2. **SD3 Medium**: https://huggingface.co/stabilityai/stable-diffusion-3-medium
   - Click "Agree and access repository"
   
3. **DeepFloyd IF**: https://huggingface.co/DeepFloyd/IF-I-XL-v1.0
   - Click "Agree and access repository"

**Important**: Use the same HuggingFace account for accepting licenses as your access token!

### 2. Create HuggingFace Token

1. Go to: https://huggingface.co/settings/tokens
2. Create new token with "read" permissions
3. Check "Read access to contents of all public gated repos you can access"
4. Save the token

### 3. Set Environment Variables

```bash
# Add to your .env file or export:
export HUGGINGFACE_HUB_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
export HF_HOME="${HOME}/.cache/huggingface"  # Or your preferred cache location
```

## 🚀 Setup Instructions

### Option 1: Direct Host Setup (Quickest)

```bash
# 1. Install PyTorch with RTX 5090 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124

# 2. Install requirements
pip install -r requirements.txt

# 3. Run setup script to download all models
python setup_models.py

# 4. Run benchmark
python research/benchmarks/efficient_benchmark_runner.py
```

### Option 2: Docker Setup (Recommended)

```bash
# 1. Build the Docker image
docker compose build benchmark

# 2. Download models first (one-time setup)
docker compose run --rm \
  -e HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN} \
  benchmark python docker_download_models.py

# 3. Run quick smoke test
docker compose run --rm \
  -e HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN} \
  benchmark python research/benchmarks/smoke_test_models.py

# 4. Run full benchmark
docker compose run --rm \
  -e HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN} \
  -e BENCHMARK_MODE=quick_sdxl_lightning \
  benchmark
```

## 📦 Model List

| Model | Repo ID | Gated | License URL |
|-------|---------|-------|-------------|
| SDXL-Turbo | stabilityai/sdxl-turbo | No | - |
| FLUX.1-dev | black-forest-labs/FLUX.1-dev | **Yes** | [Accept License](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| RealVisXL V4 | SG161222/RealVisXL_V4.0 | No | - |
| SD3 Medium | stabilityai/stable-diffusion-3-medium | **Yes** | [Accept License](https://huggingface.co/stabilityai/stable-diffusion-3-medium) |
| SDXL Base | stabilityai/stable-diffusion-xl-base-1.0 | No | - |
| HiDream-I1 | HiDream-ai/HiDream-I1-Full | No | - |
| DeepFloyd IF | DeepFloyd/IF-I-XL-v1.0 | **Yes** | [Accept License](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0) |

## 🔧 Troubleshooting

### "401 Unauthorized" Error
- ✅ Accept the model license on HuggingFace
- ✅ Wait 5-10 minutes for access to propagate
- ✅ Verify token has "read gated repos" permission
- ✅ Check you're using the same account for license and token

### "CUDA error: no kernel image available"
- ✅ Ensure using PyTorch 2.7.0+ or nightly builds
- ✅ Verify TORCH_CUDA_ARCH_LIST="12.0" is set
- ✅ Check CUDA 12.4+ is installed

### "Model not found" Error
- ✅ Model names have been fixed in the scripts
- ✅ RealVisXL is now correctly pointing to SG161222/RealVisXL_V4.0

### Docker Build Fails
- ✅ torchaudio removed (not needed for image generation)
- ✅ Using PyTorch 2.7.0 base image with CUDA 12.8

## 📊 Benchmark Modes

```bash
# Quick test (100 prompts, 1 model)
BENCHMARK_MODE=quick_sdxl_lightning

# Phase 1: TF32 validation (2 hours)
BENCHMARK_MODE=phase1

# Phase 2: Deep dive (10 hours)
BENCHMARK_MODE=phase2

# Phase 3: Resolution scaling (3 hours)
BENCHMARK_MODE=phase3

# Full benchmark (15 hours)
BENCHMARK_MODE=full
```

## 📁 File Structure

```
flexy-face/
├── research/
│   ├── benchmarks/
│   │   ├── efficient_benchmark_runner.py  # Main benchmark script (FIXED)
│   │   └── smoke_test_models.py          # Quick test script (FIXED)
│   ├── requirements.txt                   # Python dependencies
│   └── Dockerfile.benchmark               # Docker image (FIXED)
├── docker-compose.yml                     # Docker orchestration
├── setup_models.py                        # Model download script (NEW)
├── docker_download_models.py              # Docker-specific downloader (NEW)
└── .env                                   # Environment variables
```

## 🎯 Next Steps

1. **Accept all gated model licenses** (see links above)
2. **Set your HuggingFace token** in .env file
3. **Run setup_models.py** to download all models
4. **Run smoke test** to verify all models load
5. **Start benchmark** with desired mode

## ⚡ RTX 5090 Performance Expectations

With proper setup, expect:
- SDXL generation: ~15 seconds at 1024×1024 (32% faster than 4090)
- SD 1.5 generation: 5-8 seconds at 512×512
- 32GB VRAM enables larger batches and multiple models in memory
- TF32 provides ~15-20% speedup when enabled

## 📝 Notes

- Models are cached in `HF_HOME` directory (default: ~/.cache/huggingface)
- Docker mounts this cache to avoid re-downloading
- First download may take 30-60 minutes depending on connection
- Total model size: ~50-60GB

---

**Ready to benchmark!** Once you've accepted the licenses and set your token, the scripts will handle everything else.
