# Flexy Face (FLUX)

Full-stack app for FLUX text-to-image generation. Describe the style you want and the stack (FastAPI + diffusers on the backend and React + Vite on the frontend) will render it from scratch.

## Quick start with Docker

```bash
cp .env.example .env   # add your Hugging Face token if models are gated
docker compose up --build
```

Backend will be on `http://localhost:6969`, frontend on `http://localhost:6970` (served by Nginx). If you want GPU acceleration inside the backend container, run Docker with GPU support (e.g., `docker compose --profile gpu up` or set your Docker runtime to NVIDIA).

## Run locally (no Docker)

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --port 6969
```

Frontend:

```bash
cd frontend
npm install
npm run dev -- --port 6970
```

Quick test: open `http://localhost:6970`, craft a prompt (e.g. “cherry tree on a hill”), and you should see the generated image. UI is mobile-friendly: controls stack on top, output below with recent images.

## Flow

- `/api/generate`: text → image (FLUX.1 Schnell by default).
- Frontend supports prompt, steps, guidance, and optional seed.

See `docs/AGENT_INSTRUCTIONS.md` for the original build brief.
