## Next Steps (snapshot @ 2025-11-22)

We’ve aligned the backend API to the frontend contract, added session privacy, multipart trigger uploads, and heuristic+LLM prank matching. This doc captures the state so we can resume quickly.

### Backend state
- **main.py**
  - `POST /api/generate`: returns `generation_id`, `model_id`, `thumbnail_base64`, `router_metadata` (heuristics+LLM router for image model selection).
  - `POST /api/pranks`: creates prank with `share_slug` + `builder_slug`, stores `session_id`.
  - `GET /api/pranks/{slug}`: returns full prank object (triggers with base64/thumbnailBase64).
  - `GET /api/pranks?session_id=...`: lists pranks for a session (privacy: returns [] if session_id missing).
  - `POST /api/pranks/{slug}/triggers`: multipart (`sessionId`, `triggerText`, `file`), session ownership enforced if present, saves image+thumb, returns camelCase fields.
  - `GET /api/pranks/{slug}/triggers/{trigger_id}/thumbnail`: serves trigger thumbnails.
  - `POST /api/p/{slug}/generate`: uses heuristic+LLM prank matcher to decide trigger; falls back to real generation otherwise; logs match counts.
  - `GET /api/generations`: now filters by `session_id` (returns [] if missing), uses indexes on `session_id`, `created_at`.
  - `GET /api/images/thumb/{generation_id}`: serves generation thumbnails (FileResponse).
  - CORS: `allow_origins=["*"]` for now.

- **prank_matching.py**
  - Heuristic matcher (exact/substring/Jaccard) returning `(idx, scores)` with confidence threshold.
  - Optional LLM matcher (`PrankMatcherLLM`) default model env `PRANK_MATCHER_LLM_ID` (default `Qwen/Qwen2.5-1.5B-Instruct`), lazy-loaded; returns trigger index or None.
  - `match_prank_trigger` combines heuristics + LLM, returns final idx + `MatchDebug` (heuristic idx/scores, LLM idx, used_llm flag).

- **router_engine.py**
  - Image model router uses heuristics; LLM (Qwen2.5-1.5B) fallback when confidence is low. Clean fallback if LLM unavailable.

- **models.py**
  - Prank: `share_slug`, `builder_slug`, `session_id`, `view_count`.
  - PrankTrigger: `image_path`, `thumbnail_path`, `match_count`.
  - GenerationLog: includes `session_id`, indexes on `session_id` and `created_at`.

- **schemas.py**
  - CamelCase aliases for prank trigger responses, prank/prank list responses, generation responses include IDs, model_id, thumbnail_base64, router_metadata, etc.

- **storage.py**
  - `save_prank_image_with_thumbnail(slug, payload, extension)` saves full + thumb; `save_generation_image` saves full + thumb.

- **Systemd/ngrok**
  - Unit: `promptpics.service` (docker compose up; runs ngrok via `scripts/start_ngrok_promptpics.sh`). Ensure `NGROK_AUTHTOKEN` is set in the unit; restart after edits. Check with `systemctl status promptpics.service`.

### Recent fixes to remember
- Removed conflicting ID-based trigger route; only `POST /api/pranks/{slug}/triggers` remains.
- Added session filtering for generations/pranks.
- Thumbnail endpoints added for generations and prank triggers.
- Heuristic+LLM prank matcher added; default model can be overridden with `PRANK_MATCHER_LLM_ID`.

### To-do / next work
- Refine prank matcher logging: optionally attach `MatchDebug` into router_metadata for visibility.
- Tune heuristic thresholds and prompt for the LLM matcher; consider caching/instantiating the LLM globally to avoid reload per request.
- Consider exposing static URLs for images instead of base64 for performance.
- Tighten CORS to explicit origins once Replit domain is final.
- Ensure systemd is running latest image (`sudo systemctl restart promptpics.service`) and enabled at boot.

### Quick test commands
- Create prank:
  ```bash
  curl -s -X POST https://app.promptpics.ai/api/pranks \
    -H "Content-Type: application/json" \
    -d '{"sessionId":"sess_test_123","title":"Test"}'
  ```
- Upload trigger:
  ```bash
  curl -v -X POST "https://app.promptpics.ai/api/pranks/<SLUG>/triggers" \
    -F "sessionId=sess_test_123" \
    -F "triggerText=who is the cutest baby" \
    -F "file=@/path/to/image.jpg"
  ```
- List pranks for session:
  ```bash
  curl -s "https://app.promptpics.ai/api/pranks?session_id=sess_test_123" | jq
  ```
- Generate (normal):
  ```bash
  curl -s https://app.promptpics.ai/api/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt":"test image","engine":"auto","sessionId":"sess_test"}'
  ```
- Prank generate:
  ```bash
  curl -s https://app.promptpics.ai/api/p/<SLUG>/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt":"who is the cutest baby","sessionId":"sess_test"}'
  ```
