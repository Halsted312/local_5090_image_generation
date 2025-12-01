# Flexy-Face Setup Guide for RTX 5090

A comprehensive guide for running the Flexy-Face image generation API locally on an NVIDIA RTX 5090 GPU.

---

## Overview

Flexy-Face is a FastAPI-based image generation backend that supports multiple state-of-the-art diffusion models. It provides a simple REST API for text-to-image generation with intelligent model routing.

**What you'll get:**
- Local image generation API on `http://localhost:7999`
- Multiple models optimized for different use cases
- TF32 acceleration enabled by default for RTX 40/50 series
- Automatic model unloading to manage VRAM

---

## Reference Hardware

This project was developed and tested on:

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA GeForce RTX 5090 (32GB VRAM) |
| **CPU** | AMD Ryzen Threadripper 9970X (32 cores, 64 threads) |
| **RAM** | 128GB DDR5 |
| **OS** | Ubuntu 24.04.3 LTS |
| **NVIDIA Driver** | 580.105.08 |
| **CUDA** | 12.8 |
| **Docker** | 29.1.1 |

---

## Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU VRAM** | 16GB (RTX 4080) | 24GB+ (RTX 4090/5090) |
| **System RAM** | 32GB | 64GB+ |
| **Storage** | 100GB free | 200GB+ (for model cache) |
| **OS** | Ubuntu 22.04+ | Ubuntu 24.04 |

---

## Software Prerequisites

### 1. NVIDIA Driver

Install the latest NVIDIA driver (580+ recommended for RTX 5090):

```bash
# Check current driver
nvidia-smi

# If needed, install latest driver
sudo apt update
sudo apt install nvidia-driver-580
sudo reboot
```

### 2. Docker with NVIDIA Container Toolkit

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Log out and back in, then verify
docker --version

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU is accessible in Docker
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

### 3. HuggingFace Account

You need a HuggingFace account with access to gated models:

1. Create account at [huggingface.co](https://huggingface.co)
2. Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Create a token with "Read" permissions
4. Accept the license for FLUX models at [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/flexy-face.git
cd flexy-face
```

### Step 2: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set your values:

```bash
# Required: Set your HuggingFace token
HUGGINGFACE_HUB_TOKEN=hf_your_actual_token_here

# Required: Set a secure database password
POSTGRES_PASSWORD=your_secure_password_here
DATABASE_URL=postgresql+psycopg2://flexyface:your_secure_password_here@db:5432/flexyface
```

### Step 3: Start the Services

```bash
docker compose up --build
```

First run will download models (~30-50GB total). This takes 10-30 minutes depending on your connection.

### Step 4: Verify Installation

```bash
# Health check
curl http://localhost:7999/health

# Expected response:
# {"status":"ok"}
```

---

## Available Models

| Model ID | Best For | VRAM Usage | Speed |
|----------|----------|------------|-------|
| `flux_dev` | General purpose, prompt following | ~16GB | Medium |
| `flux2_dev` | Highest quality (32B params, 4-bit) | ~20GB | Slow |
| `realvis_xl` | Photorealistic portraits & faces | ~12GB | Fast |
| `sd3_medium` | Complex scenes, text rendering | ~14GB | Fast |
| `hidream_dev` | Logos, icons, text-heavy images | ~16GB | Medium |

The API automatically routes prompts to the best model, or you can specify one explicitly.

---

## API Usage

### Generate an Image

```bash
curl -X POST http://localhost:7999/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a photorealistic cat wearing sunglasses, studio lighting",
    "width": 1024,
    "height": 1024
  }' | jq -r '.image_base64' | base64 -d > output.png
```

### Generate with Specific Model

```bash
curl -X POST http://localhost:7999/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "portrait of a woman, natural lighting",
    "model_id": "realvis_xl",
    "width": 1024,
    "height": 1024
  }' | jq -r '.image_base64' | base64 -d > portrait.png
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description of the image |
| `model_id` | string | auto | Model to use (see table above) |
| `width` | int | 1024 | Image width (256-2048) |
| `height` | int | 1024 | Image height (256-2048) |
| `num_inference_steps` | int | varies | Diffusion steps (more = better quality, slower) |
| `guidance_scale` | float | varies | How closely to follow the prompt |
| `seed` | int | random | For reproducible results |

### Response Format

```json
{
  "image_base64": "iVBORw0KGgo...",
  "model_id": "flux_dev",
  "seed": 12345,
  "duration_ms": 8500,
  "router_decision": {
    "model_id": "flux_dev",
    "reason": "General purpose prompt, good match for FLUX"
  }
}
```

---

## Performance Tips

### TF32 Acceleration

TF32 is enabled by default for RTX 40/50 series GPUs, providing ~2.7x speedup. This is configured in `docker-compose.yml`:

```yaml
environment:
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Model Unloading

The API automatically unloads models not in use to free VRAM. When you request a different model, the current one is unloaded first. This means:

- First generation with a model: ~15-30 seconds (loading)
- Subsequent generations: ~5-15 seconds
- Switching models: adds loading time

### Queue Management

The API uses a single-threaded queue to prevent GPU memory conflicts. If multiple requests come in, they're processed sequentially.

---

## Troubleshooting

### Out of Memory (OOM) Errors

```
CUDA out of memory
```

**Solutions:**
- Reduce image dimensions (try 768x768 instead of 1024x1024)
- Use a lighter model (`realvis_xl` uses less VRAM)
- Restart the container to clear VRAM: `docker compose restart backend`

### Models Not Downloading

```
401 Client Error: Unauthorized
```

**Solutions:**
- Verify your `HUGGINGFACE_HUB_TOKEN` in `.env`
- Make sure you accepted the model license on HuggingFace
- Check token has "Read" permissions

### GPU Not Detected

```
CUDA not available
```

**Solutions:**
- Verify NVIDIA driver: `nvidia-smi`
- Verify container toolkit: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi`
- Restart Docker: `sudo systemctl restart docker`

### Port Already in Use

```
bind: address already in use
```

**Solutions:**
- Check what's using port 7999: `lsof -i :7999`
- Stop conflicting service or change port in `docker-compose.yml`

---

## Running Without Docker (Advanced)

If you prefer to run without Docker:

```bash
cd backend

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set environment variables
export HUGGINGFACE_HUB_TOKEN=hf_your_token
export DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/flexyface

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 7999
```

Note: You'll need to set up PostgreSQL separately.

---

## Stopping the Services

```bash
# Stop containers (keeps data)
docker compose down

# Stop and remove all data (including model cache)
docker compose down -v
```

---

## Support

For issues and questions, open an issue on GitHub.
