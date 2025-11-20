# FLUX Backend (FastAPI)

FastAPI service wrapping the FLUX pipelines for text-to-image and image editing.

## Setup (local)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# If FluxPipeline / FluxKontextPipeline are missing:
# pip install -U git+https://github.com/huggingface/diffusers.git
```

Set your Hugging Face token if the FLUX models are gated:

```bash
export HUGGINGFACE_HUB_TOKEN=hf_your_token
```

Run the API:

```bash
uvicorn app.main:app --reload --port 8000
```

## Endpoints

- `GET /health` – health check.
- `POST /api/generate` – JSON body (`TextGenerateRequest`) for text-to-image. Returns `{ "image_base64": "..." }`.
- `POST /api/edit` – `multipart/form-data` with `file` (image) and `prompt`. Returns `{ "image_base64": "..." }`.

Pipeline defaults target CUDA; CPU is used as a fallback, but performance will degrade.
