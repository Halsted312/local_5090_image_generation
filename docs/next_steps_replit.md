# Next Steps: Replit Frontend + API Only

Context: Backend runs locally on port 7999 (FastAPI + FLUX + prank endpoints), exposed via ngrok at `https://app.promptpics.ai`. Frontend will be built in Replit at `https://promptpics.ai`.

## 1) Replit frontend wiring
- API base: `https://app.promptpics.ai`
- CORS already allows:
  - `https://promptpics.ai`
  - `https://*.replit.app`
  - `http://localhost:3000`
- Endpoints to consume:
  - `POST /api/generate` (body: `{ prompt, num_inference_steps, guidance_scale, width, height, seed? }`)
  - `POST /api/pranks` (body: `{ title?, slug? }`)
  - `POST /api/pranks/{prank_id}/triggers` (multipart: `trigger_text`, `file`)
  - `POST /api/p/{slug}/generate` (body: `{ prompt, num_inference_steps?, guidance_scale?, width?, height?, seed? }`)
  - `GET /api/pranks/{slug}` (returns title + triggers with base64 images)

## 2) Ngrok/Domain
- `~/.config/ngrok/ngrok.yml` upstream is set to `7999` (API). If you change ports, update and restart ngrok.
- Tunnel is `https://app.promptpics.ai -> http://localhost:7999`.

## 3) Systemd (optional)
- Use `docs/systemd-promptpics.md` to auto-start docker + ngrok on boot.
- Services: `promptpics-docker.service`, `promptpics-ngrok.service`.

## 4) Dev/local testing
- `docker compose up backend db` (frontend is archived under `frontend_archive_ignore/`).
- Health: `http://localhost:7999/health`.
- Postman/curl against `http://localhost:7999/api/...` or via ngrok URL if you need external access.

## 5) Known constraints / TODOs
- GPU OOM: defaults are 640×640; adjust request sizes if needed.
- Prank slugs: default 5 chars; can be set 3–16 chars on create.
- Admin: `/create/admin` logic lives in archived frontend; Replit must implement its own UI for `/imagine` using the APIs above.
