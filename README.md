# Flexy-Face

A local image generation API powered by FLUX and other state-of-the-art diffusion models. Designed for NVIDIA RTX 5090/4090 GPUs.

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/YOUR_USERNAME/flexy-face.git
cd flexy-face
cp .env.example .env
# Edit .env with your HuggingFace token and database password

# 2. Start services
docker compose up --build

# 3. Test
curl http://localhost:7999/health
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed installation instructions.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/generate` | POST | Generate image from text prompt |
| `/api/models` | GET | List available models |

## Generate an Image

```bash
curl -X POST http://localhost:7999/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat wearing sunglasses"}' \
  | jq -r '.image_base64' | base64 -d > cat.png
```

## Available Models

| Model | Best For |
|-------|----------|
| `flux_dev` | General purpose |
| `flux2_dev` | Highest quality (32B 4-bit) |
| `realvis_xl` | Photorealistic portraits |
| `sd3_medium` | Complex scenes, text |
| `hidream_dev` | Logos, icons |

## Requirements

- NVIDIA GPU with 16GB+ VRAM (RTX 4080/4090/5090)
- Docker with NVIDIA Container Toolkit
- HuggingFace account with FLUX access

## Prank Links

This project includes an optional "prank link" feature where certain prompts can return pre-set images. See the API for `/api/pranks` endpoints.

## License

MIT
