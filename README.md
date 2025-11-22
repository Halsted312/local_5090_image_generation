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

````markdown
# PromptPics / PromptPix Project Summary  
*(for future you + future AI agents)*

This document is a **complete, technical summary** of what we’ve built and planned in this chat, so you can hand it to another chatbot (Claude, another ChatGPT session, etc.) and continue smoothly.

---

## 0. High-Level Concept

**PromptPics / PromptPix** is an AI image app with two faces:

1. **Serious AI image generator:**
   - Users type prompts → your GPU (Flux model) generates images.
   - Optionally upload an image and apply edits.

2. **“Prank link” super-mode:**
   - You (the admin) create **secret links**.
   - For normal prompts, the link behaves like a regular AI generator.
   - For certain “trap prompts” you set (e.g., “most beautiful girl in the world”), a pre-uploaded image is returned instead (e.g., a friend’s photo).

The final architecture you chose (Option C):

- **Front-end SPA** at `https://promptpics.ai` hosted on **Replit**.
- **Backend API** (FastAPI + FLUX + prank logic) running on your **desktop with an RTX 5090**, exposed via **ngrok** at `https://app.promptpics.ai`.
- **Systemd + Docker** on your desktop so everything auto-starts at boot.

---

## 1. Backend: FLUX GPU Image API on Your Desktop

### 1.1 Core Tech Stack

- **Language:** Python 3
- **Framework:** FastAPI
- **Models:** FLUX text-to-image and Kontext (image editing) via 🤗 diffusers
- **GPU:** RTX 5090, using CUDA, bfloat16, CPU offload if needed
- **Data models:** Pydantic for request/response
- **Image processing:** PIL (Pillow)
- **Server runtime:** Uvicorn (behind Docker)

#### FLUX configuration helper (`config.py`)

- Picks **model IDs** for text and Kontext models via env vars:
  - `FLUX_TEXT_MODEL_ID` (default `black-forest-labs/FLUX.1-schnell`)
  - `FLUX_KONTEXT_MODEL_ID` (default `black-forest-labs/FLUX.1-Kontext-dev`)
- Selects device: CUDA if available, otherwise CPU.  
  :contentReference[oaicite:0]{index=0}  

Key logic (summarized):

```python
# Pseudocode
def get_device() -> Literal["cuda", "cpu"]:
    if DEVICE or FORCE_DEVICE == "cpu": return "cpu"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"
````

#### Lazy-loaded FLUX pipelines (`flux_models.py`)

* Maintains **singleton** instances of:

  * `FluxPipeline` for **text → image**
  * `FluxKontextPipeline` for **image editing**
* Uses `get_device()` to determine GPU vs CPU.
* Uses `HUGGINGFACE_HUB_TOKEN` to authenticate Hugging Face downloads.
* Configures:

  * bfloat16 dtype
  * attention/vae slicing
  * model CPU offload if GPU OOM happens


Pseudocode structure:

```python
# _load_text_pipeline:
device = get_device()
token = HF_TOKEN
pipeline = FluxPipeline.from_pretrained(FLUX_TEXT_MODEL_ID, torch_dtype=torch.bfloat16, token=token)
pipeline.to(device) or enable_model_cpu_offload()
pipeline.enable_attention_slicing()
...

# get_text_pipeline() uses double-checked locking to ensure singleton.
```

---

### 1.2 FastAPI Application & Core Endpoints

The main backend app is defined in `main.py` using FastAPI. 

#### CORS configuration

Initially, CORS allowed only localhost origins (for your local React app):

```python
ALLOWED_ORIGINS: Iterable[str] = (
    "http://localhost:6970",
    "http://127.0.0.1:6970",
    "http://localhost",
    "http://127.0.0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

For Option C, we planned to **extend this** to:

* `https://promptpics.ai` (Replit front-end)
* Replit preview domains (e.g. `https://*.replit.app`)
* Optional local dev origin (e.g. `http://localhost:3000`)

CORS best practice here is to list your exact front-end origins and allow all methods/headers.

#### Data models (`schemas.py`)

You defined two Pydantic models: 

```python
class TextGenerateRequest(BaseModel):
    prompt: str
    num_inference_steps: int = Field(4, ge=1, le=50)
    guidance_scale: float = Field(0.0, ge=0.0, le=50.0)
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    seed: int | None = Field(None)

class ImageResponse(BaseModel):
    image_base64: str
```

#### `/health` endpoint

Simple health check: returns `{"status": "ok"}`. 

#### `/api/generate` – Text-to-image FLUX

* Accepts `TextGenerateRequest`.
* Calls `get_text_pipeline()` from `flux_models`.
* Creates an optional seeded `torch.Generator`.
* Runs the FLUX pipeline with:

  * `prompt`
  * `num_inference_steps`
  * `guidance_scale`
  * `width`, `height`
  * `generator`
* Takes `result.images[0]`, encodes as PNG → base64 string.
* Returns an `ImageResponse(image_base64=...)`. 

#### `/api/edit` – Kontext image editing

* Accepts:

  * `file: UploadFile` (image)
  * `prompt: str` (Form)
  * `num_inference_steps: int` (Form, default 28)
  * `guidance_scale: float` (Form, default 3.5)
  * `seed: int | None` (Form)
* Validates `file.content_type` starts with `"image/"`.
* Opens the image via PIL, converts to RGB.
* Calls `get_kontext_pipeline()` and runs the edit pipeline with:

  * `image=init_image`
  * `prompt`
  * `num_inference_steps`
  * `guidance_scale`
  * `generator` (seeded)
* Returns edited image as base64 PNG. 

---

### 1.3 Planned Prank / “Trick” API (not fully coded yet)

We designed a **prank system architecture** to support the “fake generation” behavior.

#### Data model (Postgres, conceptual)

Tables:

```sql
CREATE TABLE pranks (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  slug VARCHAR(16) UNIQUE NOT NULL,
  title TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE prank_triggers (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  prank_id UUID NOT NULL REFERENCES pranks(id) ON DELETE CASCADE,
  trigger_text TEXT NOT NULL,
  image_path TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

* `pranks.slug` is used in URLs: `/p/<slug>`.
* `prank_triggers` lists each trap prompt + prank image.

#### Endpoints (spec)

We designed:

1. `POST /api/pranks`

   * Request: `{ "title": string | null }`
   * Response: `{ prank_id, slug, share_url }`
   * Creates a prank with a random slug (e.g. 8 chars).

2. `POST /api/pranks/{prank_id}/triggers`

   * Form:

     * `trigger_text` (str)
     * `file` (image)
   * Saves image to disk (e.g. `./prank_images/<slug>/<uuid>.png`).
   * Inserts a row in `prank_triggers`.

3. `POST /api/p/{slug}/generate`

   * Request: `{ "prompt": "user's prompt" }`
   * Logic:

     * Look up prank by `slug`.
     * Load its triggers.
     * Use a **local 8B LLM** (Llama 3-8B Instruct) to decide if the prompt is “essentially the same” as any trigger.

       * We wrote a JSON-only classifier prompt:
         `"Return {\"match\": true|false, \"index\": int|null}"`
         for a list of trap prompts.
     * If match:

       * Return the prank image from disk (converted to base64 PNG).
     * Else:

       * Call `/api/generate` internally and return a real FLUX image.

The classification is done via a small local LLM loaded with `transformers` and `pipeline("text-generation")`, with low temperature and short max tokens.

---

## 2. Front-end: Local React App (“Imgen 4 U”)

Originally, your front-end lived on your desktop as a React app (Vite). We’ll re-use this design for the Replit SPA.

### 2.1 Stack

* **React 18**
* **TypeScript**
* **Vite** (implied by `main.tsx` and file structure)
* Styling with **plain CSS** (no Tailwind).

### 2.2 Component Layout

#### `App.tsx` – Main UI container

* Manages:

  * `mode`: `"generate"` or `"edit"`
  * `prompt` (default `"show me a cherry tree on a hill"`)
  * `steps` slider (1–10)
  * `guidance` slider (1–3)
  * `file` (image for edit mode)
  * `images`: list of generated/edited images
  * `isLoading`, `error`

* Handles mapping UI sliders to backend parameters:

  * Generate:

    * `num_inference_steps ≈ 4 + (steps-1)*0.5` (≈4–8.5)
    * `guidance_scale = guidance === 1 ? 0 : guidance - 1`
  * Edit:

    * `num_inference_steps ≈ 10 + (steps-1)*3` (clamped to <=40)
    * `guidance_scale = 1 + (guidance-1)*1.5`



* Uses `generateImage` and `editImage` functions from `api.ts` (not included in files but implied) to call the FastAPI endpoints.

* Layout:

  * Header with title “Imgen 4 U” and description.
  * Two main panels in a grid:

    * Left: mode toggle + `PromptForm`.
    * Right: `ImageViewer`.

#### `PromptForm.tsx` – Form and sliders

* Controlled inputs:

  * `textarea` for prompt.
  * Two sliders: “Detail” and “Guidance”.
* In **edit mode**, shows:

  * “Choose an image” button (file input behind a styled label).
  * Selected file name.
  * Preview thumbnail (via `URL.createObjectURL`).
* Submits via `onSubmit()` callback. 

#### `ImageViewer.tsx` – Output gallery

* Takes `images: GeneratedImage[]`.
* If empty: “No images yet — craft a prompt and go.”
* Otherwise:

  * Renders `image-card`s: `<img>` with base64 PNG data URI, plus a pill badge (“Generated” or “Edited”) and the prompt. 

#### `index.css` – Visual design

* Uses `Space Grotesk / Inter` style fonts.
* Dark, gradient background.
* Panels with glassmorphism (semi-transparent, rounded corners, drop shadows).
* Responsive grid:

  * 2 columns on desktop (`.layout`).
  * 1 column stack on mobile (`@media (max-width: 900px)`).
* Styled sliders with custom thumbs/track, nice badges, and CTA buttons. 

#### `main.tsx`

Standard React root render:

````ts
ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
``` :contentReference[oaicite:13]{index=13}  

---

## 3. Networking: Domains, ngrok, and GoDaddy

### 3.1 Domains

You currently have:

- **`promptpix.ai`** – older domain.
- **`promptpics.ai`** – new, preferred brand domain.
- **`app.promptpics.ai`** – ngrok custom domain (subdomain) pointing to your desktop box via CNAME from GoDaddy to an `ngrok-cname.com` target.

Because GoDaddy **does not allow CNAME at the apex/root**, custom domains with ngrok must be on a **subdomain** (e.g. `app`, `www`).   

### 3.2 ngrok configuration

Using ngrok v3 with a YAML config at `~/.config/ngrok/ngrok.yml`:

Initial (when front-end was on 7080):

```yaml
version: 3

agent:
  authtoken: <YOUR_NGROK_AUTHTOKEN>

endpoints:
  - name: promptpics
    url: https://app.promptpics.ai
    upstream:
      url: 7080   # old: Nginx frontend
````

For Option C (backend API only), we changed `upstream.url` → **7999**:

```yaml
endpoints:
  - name: promptpics
    url: https://app.promptpics.ai
    upstream:
      url: 7999   # FastAPI backend
```

Common commands:

* Quick test: `ngrok http 7999 --url=https://app.promptpics.ai`
* Config-based: `ngrok start promptpics`

You hit an error `ERR_NGROK_334` (“endpoint already online”) when:

* You had a manual ngrok session running *and* systemd trying to start the same endpoint.
* Fixed by killing stray ngrok processes (`pkill ngrok`) and letting systemd own the tunnel.

ngrok custom domain + CNAME pattern is consistent with ngrok’s official docs.

---

## 4. Systemd + Docker Orchestration

You created two systemd units to autoboot your stack:

1. `promptpics-docker.service`
2. `promptpics-ngrok.service`

### 4.1 `promptpics-docker.service`

Purpose: run `docker compose up -d` and keep containers up across reboots.

```ini
# /etc/systemd/system/promptpics-docker.service
[Unit]
Description=PromptPics docker compose stack
After=network-online.target docker.service
Wants=network-online.target docker.service

[Service]
Type=oneshot
WorkingDirectory=/home/halsted/Python/flexy-face
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
RemainAfterExit=yes
TimeoutStartSec=0
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

This starts containers:

* `flexy-face-db-1`
* `flexy-face-backend-1` (FastAPI)
* `flexy-face-frontend-1` (old UI, keep as dev-only if you want).

### 4.2 `promptpics-ngrok.service`

Purpose: keep the `promptpics` ngrok endpoint running.

```ini
# /etc/systemd/system/promptpics-ngrok.service
[Unit]
Description=Ngrok tunnel for PromptPics
After=network-online.target promptpics-docker.service
Wants=promptpics-docker.service

[Service]
Type=simple
ExecStart=/home/halsted/Python/flexy-face/scripts/start_ngrok_promptpics.sh
Restart=on-failure
User=halsted
WorkingDirectory=/home/halsted/Python/flexy-face
Environment=NGROK_CONFIG=/home/halsted/.config/ngrok/ngrok.yml

[Install]
WantedBy=multi-user.target
```

And the script:

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "Starting ngrok tunnel 'promptpics' (app.promptpics.ai → localhost:7080)..."
/usr/local/bin/ngrok start promptpics
```

For Option C you’ll update the log message and ensure upstream is 7999.

Enable + start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable promptpics-docker.service promptpics-ngrok.service
sudo systemctl start promptpics-docker.service promptpics-ngrok.service
```

Now:

* System boots → docker stack comes up → ngrok tunnel starts → `app.promptpics.ai` is live.

---

## 5. Final Architecture (Option C): Replit SPA + Desktop API

This is the design you ultimately chose.

### 5.1 High-level view

* **Front-end SPA on Replit**:

  * Domain: `https://promptpics.ai`
  * React + TypeScript.
  * Routes: `/` (landing + generator) and `/imagine` (super mode / prank builder).
  * Uses `fetch()` to call `https://app.promptpics.ai/api/...`.

* **Backend API on your desktop**:

  * Domain: `https://app.promptpics.ai` (ngrok custom domain).
  * FastAPI + FLUX + prank logic.
  * Exposes:

    * `POST /api/generate`
    * `POST /api/edit`
    * `POST /api/pranks`
    * `POST /api/pranks/{prank_id}/triggers`
    * `POST /api/p/{slug}/generate`
  * CORS allows `https://promptpics.ai`.

This mirrors a common SPA + backend architecture where the front-end is static on one host and the backend is a remote API.

---

### 5.2 API client stub for the Replit SPA

You’ll have something like this in `src/api.ts`:

```ts
const API_BASE = import.meta.env.VITE_API_BASE_URL || "https://app.promptpics.ai";

export async function generateImage(payload: {
  prompt: string;
  num_inference_steps: number;
  guidance_scale: number;
  width: number;
  height: number;
}): Promise<string> {
  const res = await fetch(`${API_BASE}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("Generation failed");
  const data = await res.json();
  return data.image_base64;
}

export async function editImage(file: File, prompt: string, opts: { num_inference_steps: number; guidance_scale: number }): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  form.append("prompt", prompt);
  form.append("num_inference_steps", String(opts.num_inference_steps));
  form.append("guidance_scale", String(opts.guidance_scale));
  const res = await fetch(`${API_BASE}/api/edit`, { method: "POST", body: form });
  if (!res.ok) throw new Error("Edit failed");
  const data = await res.json();
  return data.image_base64;
}

// Prank APIs
export async function createPrank(title?: string) {
  const res = await fetch(`${API_BASE}/api/pranks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw new Error("Failed to create prank");
  return res.json() as Promise<{ prank_id: string; slug: string; share_url: string }>;
}

export async function addPrankTrigger(prankId: string, triggerText: string, file: File): Promise<void> {
  const form = new FormData();
  form.append("trigger_text", triggerText);
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/pranks/${prankId}/triggers`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error("Failed to add prank trigger");
}

export async function generatePrankImage(slug: string, prompt: string): Promise<string> {
  const res = await fetch(`${API_BASE}/api/p/${slug}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) throw new Error("Prank generation failed");
  const data = await res.json();
  return data.image_base64;
}
```

---

### 5.3 Replit front-end SPA design brief

**Tell your Replit agent:**

> Build a React + TypeScript SPA (Single Page App) with React Router, deployed as a static app on `promptpics.ai`, with two routes:
>
> * `/` → landing page + chat/image generator
> * `/imagine` → super mode for creating prank links
>
> Use `VITE_API_BASE_URL` (pointing to `https://app.promptpics.ai`) to talk to the backend API.

#### `/` – Landing + generator

Sections:

1. **Hero:**

   * Big title: “PromptPics”
   * Tagline: “Turn any idea into an image in seconds.”
   * CTA button: scrolls to the generator section.

2. **Generator:**

   * Prompt textarea.
   * Sliders:

     * “Detail” (1–10) → maps to `num_inference_steps`.
     * “Guidance” (1–3) → maps to `guidance_scale`.
   * Toggle: “Generate from text” / “Edit uploaded photo”.
   * Upload image option when in edit mode.
   * “Generate” button → calls `generateImage` or `editImage`.
   * Output panel reused from `ImageViewer` logic: show generated images with their prompts.

3. **Explanation / marketing:**

   * 2–3 cards explaining:

     * “Write a prompt” (core feature).
     * “Upload + transform photos”.
     * “Create secret prank links in Super Mode”.

4. **Footer:**

   * Link to `/imagine` (“Super Mode”)
   * Maybe a GitHub link, your name, etc.

#### `/imagine` – Super Mode prank builder

* Title: “Super Mode: Prank Links”
* Flow:

  1. Input for prank title.
  2. Button: “Create prank link” → calls `createPrank`.
  3. Show result:

     * `Prank ID`
     * `Share URL` (e.g. `https://promptpics.ai/p/<slug>` or `https://app.promptpics.ai/p/<slug>`)
     * Copy button.
  4. Below: triggers list:

     * For each row:

       * Textarea for trigger prompt.
       * File upload for prank image.
       * Preview thumbnail.
       * “Add trigger” button → calls `addPrankTrigger`.
* A short explanation of how the link behaves.

The **prank runtime page** `/p/:slug` can either:

* be a front-end route in the SPA that calls `generatePrankImage(slug, prompt)`, or
* be served by your backend directly (if you later choose SSR).

For Option C, easiest is to make `/p/:slug` another React route that shows a cut-down version of the main generator UI but calls the prank API instead of the plain generator.

---

### 5.4 Analytics (GA4)

Because everything user-facing is on `promptpics.ai`:

1. Set up **one GA4 property** for `promptpics.ai`.
2. Add GA4 snippet or GTM to the SPA.
3. For route changes (`/` ↔ `/imagine` ↔ `/p/:slug`), manually send GA4 “virtual pageviews” or configure GA4’s enhanced measurement for SPAs.
4. Track events:

   * `generate_click` (with metadata like mode: generate/edit, prompt length, maybe tags).
   * `prank_create`, `prank_trigger_add`.
   * `prank_generate` for friend usage.

---

## 6. Chronological Timeline (What Happened in This Chat)

To help another AI “time travel” through your journey, here’s the story:

1. You started with a **local FLUX image generator**:

   * FastAPI backend (`/api/generate`, `/api/edit`) using FLUX pipelines.
   * React front-end with a nice UI (Imgen 4 U) running on your desktop.

2. You came up with the **prank idea**:

   * Secret URLs `/p/<slug>` where some prompts trigger prank images and others generate real images.
   * We initially designed a **vector embedding** approach for matching prompts.
   * You then decided you preferred a **small local LLM (8B)** to decide “are these prompts the same?”, so we designed a JSON-only classification prompt + `choose_matching_trigger()`.

3. You bought domains (`promptpix.ai`, then `promptpics.ai`) and wanted everything to be reachable from a real domain.

4. We wired **ngrok** + **GoDaddy**:

   * Created `app.promptpics.ai` as a reserved domain in ngrok.
   * Added CNAME in GoDaddy: `app` → ngrok’s CNAME target.
   * Configured `~/.config/ngrok/ngrok.yml` with an endpoint named `promptpics`.

5. We automated your desktop stack with **Docker + systemd**:

   * `promptpics-docker.service` to run `docker compose up -d`.
   * `promptpics-ngrok.service` to run `start_ngrok_promptpics.sh`.
   * Fixed path issues (`/usr/local/bin/ngrok`) and conflicts (`ERR_NGROK_334`).

6. You then decided you wanted a **proper landing page and brand** on `promptpics.ai`, and liked the idea of Option C:

   * Replit hosts the **SPA** (landing + app).
   * Your desktop is **just the API**, hidden behind ngrok.
   * Better Google Analytics + user experience.

7. We designed Option C in detail:

   * CORS rules.
   * API client stubs.
   * React/Router structure (`/` and `/imagine`).
   * GA4 events + routes.
   * Domain roles: `promptpics.ai` as the main, `app.promptpics.ai` for API.

---

## 7. How to Use This Document with Another Chatbot

When you open a new chat with Claude or a fresh ChatGPT, paste something like:

> “Here is a detailed Markdown spec of my current project (PromptPics). Please read it fully and then help me with [specific task]…”

And include this entire document.

Then you can ask for:

* Front-end help: “Port this UI to Replit React SPA at promptpics.ai”.
* Back-end help: “Implement the prank endpoints as described using FastAPI and SQLAlchemy”.
* DevOps help: “Harden my systemd services and add log rotation”.
* Analytics help: “Instrument GA4 for this SPA.”
