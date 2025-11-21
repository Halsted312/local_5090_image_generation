# Flexy Face (FLUX)

Full-stack app for FLUX text-to-image generation. Describe the style you want and the stack (FastAPI + diffusers on the backend and React + Vite on the frontend) will render it from scratch.

## Quick start with Docker

```bash
cp .env.example .env   # add your Hugging Face token if models are gated
docker compose up --build
```

Backend will be on `http://localhost:7999`, frontend on `http://localhost:7080` (served by Nginx). A Postgres container is included for prank metadata, and `./prank_images` is bind-mounted into the backend at `/data/prank_images`. If you want GPU acceleration inside the backend container, run Docker with GPU support (e.g., `docker compose --profile gpu up` or set your Docker runtime to NVIDIA).

Expose via `app.promptpics.ai` (ngrok):
- Reserve `app.promptpics.ai` in the ngrok dashboard and add the provided CNAME in GoDaddy pointing to your domain.
- Create `~/.config/ngrok/ngrok.yml` as shown in `docs/ngrok-promptpics.md` (tunnel named `promptpics` → localhost:7999).
- Start the backend, then run `./scripts/start_ngrok_promptpics.sh`.
- Visit `https://app.promptpics.ai` once DNS has propagated.

## Run locally (no Docker)

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --port 7999
```

Frontend:

```bash
cd frontend
npm install
npm run dev -- --port 7080
```

Quick test: open `http://localhost:7080`, craft a prompt (e.g. “cherry tree on a hill”), and you should see the generated image. UI is mobile-friendly: controls stack on top, output below with recent images.

## Flow

- `/api/generate`: text → image (FLUX.1 Schnell by default).
- `/api/p/{slug}/generate`: prank endpoint – will return a stored prank image if the prompt matches a trap, otherwise falls back to FLUX.
- `/api/pranks` and `/api/pranks/{prank_id}/triggers`: operator-only helpers to register prank slugs and attach stored images (no public uploads).
- Place prank assets under `./prank_images/<slug>/...` on the host; triggers reference paths relative to that folder.

See `docs/AGENT_INSTRUCTIONS.md` for the original build brief.
