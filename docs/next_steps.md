# Next Steps (FLUX App)

## What’s in place
- FastAPI backend with FLUX text and Kontext pipelines (bf16 on CUDA; CPU offload fallback to avoid OOM).
- Frontend (Vite/React) with mobile-friendly layout: text vs photo modes, prompt defaults, sliders (Detail 1–10, Guidance 1–3), upload + previews, fixed-action bar.
- Dockerized stack: backend on 7999, frontend on 7080; HF cache volume persisted.
- Verified: `/api/generate` 200 on 512×512 test; Kontext edits working after bf16/offload changes.

## How we tuned it
- GPUs: PyTorch 2.7.0 + CUDA 12.8 (sm_120 support for RTX 5090).
- Pipelines: bfloat16, `.to("cuda")` with attention/vae slicing; CPU offload fallback on OOM.
- Edit defaults: 28 steps, guidance 3.5; UI maps Detail/Guidance to sane ranges for text vs edits.
- CORS: backend allows localhost:7080 by default (override with `CORS_ALLOW_ORIGINS`).

## Immediate next steps
1) **Ngrok/Internet access**  
   - Add an `ngrok.yml` or simple script to tunnel port 7080 (frontend) and/or 7999 (API).  
   - Example: `ngrok http 7080` (ensure auth token set: `ngrok config add-authtoken <token>`).  
   - Optional: add a `docs/ngrok.md` with the exact commands and any callback URLs you need.
2) **Stability/VRAM guardrails**  
   - Add a lightweight health check to confirm pipelines are loaded (store a flag and skip reloading).  
   - Expose a “reset pipeline” endpoint to release VRAM if needed.  
   - Cap edit resolution in the UI (e.g., 1024×1024) to avoid spikes.
3) **Prompt/slider presets**  
   - Surface presets for edits (e.g., “keep identity, change expression”, Guidance 2.5, Detail 7).  
   - Optionally pin text guidance to 0 by default and expose it only in an “advanced” drawer.
4) **Backup model**  
   - Add SDXL or a smaller edit model as a fallback (`/api/edit_sdxl`) for comparison and resilience.
5) **Persistence/logging**  
   - Save last N generations to disk with prompt metadata for debugging.  
   - Add structured logging around pipeline load times and OOM fallbacks.

## How to run (recap)
```bash
docker compose up -d        # backend:7999, frontend:7080
# Frontend URL: http://localhost:7080
```

If you need to force CPU (debug only): `DEVICE=cpu docker compose up -d`.
