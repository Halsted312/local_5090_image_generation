Short answer: yes, FLUX is **exactly** the kind of “better” model your hardware was built for, and yes, you can absolutely support both:

* **Text → Image** (prompt → new image)
* **Image + Text → Edited Image** (upload + “make it sunset”, etc.)

I’ll give you:

1. **Quick FLUX overview & why it fits your 5090**
2. **What the backend should look like (FastAPI + FLUX)**
3. **What the frontend should look like (prompt + upload)**
4. **A big, copy-paste prompt for your VS Code agent (with FLUX + edit support)**

---

## 1. Is FLUX good for you?

**Yes. FLUX.1 is SOTA open-weight** from the original Stable Diffusion team (Black Forest Labs). It’s a 12B-parameter diffusion transformer family (Dev, Schnell, Kontext, Fill, etc.) that’s being treated as the next step beyond SDXL in the open-source world.

Key points:

* **FLUX.1 [schnell]** – text→image, optimized for speed (high quality in 1–4 steps, Apache-2.0 licensed).
* **FLUX.1 [dev]** – text→image, maximum quality open weights, non-commercial license.
* **FLUX.1 Kontext [dev]** – unified **image editing + text-to-image**; take an image + instruction, and it edits with strong consistency.
* **FLUX.1 Fill [dev]** – **inpainting/outpainting** (image + mask + text).

VRAM / system:

* FLUX Dev/Schnell are big: recommended **24 GB+ VRAM** for FP16/BF16. Several guides call out 24 GB as the “proper” tier and explicitly recommend the RTX 5090 as a top choice for FLUX.
* Community has run them on **14–16 GB** with quantization and careful settings, but you’re way beyond that with your 5090 (32 GB) + 128 GB RAM.

So: your hardware is basically “dream tier” for FLUX.1 Schnell/Dev + Kontext/Fill.

---

## 2. Backend design with FLUX (text→image + edit)

You already liked the FastAPI + Python setup, so let’s keep that and just **swap SDXL for FLUX**.

### 2.1 Back-end components

Use Hugging Face **diffusers** – it has dedicated FLUX pipelines:

* `FluxPipeline` – text→image.
* `FluxKontextPipeline` – image editing + text→image.
* `FluxFillPipeline` – inpainting/outpainting with image + mask.

You’ll need a recent `diffusers` (≥ 0.32, or main branch, depending on when you install) and BF16/FP16 capable PyTorch.

**Environment (backend)**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate

pip install --upgrade "torch==2.3.1"  # or latest CUDA build that matches your drivers
pip install --upgrade diffusers transformers accelerate safetensors sentencepiece
# if Flux pieces are missing:
# pip install -U git+https://github.com/huggingface/diffusers.git
```

And you’ll need a Hugging Face login + accepted terms for FLUX models (they’re gated, but open weights).
Set `HUGGINGFACE_HUB_TOKEN` as an env var for the backend process if needed.

### 2.2 Text → image with FLUX.1 [schnell]

Minimal pattern from HF docs:

```python
from diffusers import FluxPipeline
import torch

model_id = "black-forest-labs/FLUX.1-schnell"  # or FLUX.1-dev

pipe = FluxPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
# pipe.enable_model_cpu_offload()  # optional if you want to save VRAM
```

Then call:

```python
image = pipe(
    prompt="a cinematic futuristic city, 8k, volumetric lighting",
    num_inference_steps=4,
    guidance_scale=0.0,  # FLUX often uses 0 or small guidance
    height=1024,
    width=1024,
).images[0]
```

### 2.3 Image editing with FLUX.1 Kontext

Kontext pipeline from the HF card:

```python
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import torch

pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

init_image = load_image("path/to/uploaded.png")
edited = pipe(
    image=init_image,
    prompt="turn this into a rainy cyberpunk night scene",
    num_inference_steps=4,
    guidance_scale=0.0,
).images[0]
```

For **inpainting/outpainting** with masks, you’d use `FluxFillPipeline` (image + mask + prompt).

### 2.4 API shape I’d use

FastAPI app with **two core endpoints**:

* `POST /api/generate` – text→image (FluxPipeline)
* `POST /api/edit` – image edit (FluxKontextPipeline)

  * Accepts multipart form: `file` (image), `prompt` (instruction), optional `mode`/params

**/api/generate**

Request JSON:

```json
{
  "prompt": "a retro-futuristic skyline at dusk",
  "num_inference_steps": 4,
  "guidance_scale": 0.0,
  "width": 1024,
  "height": 1024,
  "seed": 12345
}
```

Response JSON:

```json
{
  "image_base64": "<base64-png>"
}
```

**/api/edit**

`multipart/form-data`:

* `file`: uploaded PNG/JPEG
* `prompt`: “turn this daytime photo into a cinematic sunset with dramatic clouds”
* optional: `num_inference_steps`, `guidance_scale`, etc.

Response: same `{ "image_base64": "..." }`.

Your FastAPI route would look a lot like the SDXL example from before, but now calling `FluxPipeline`/`FluxKontextPipeline` and returning base64 PNG bytes.

---

## 3. Frontend: prompt + upload, minimal but powerful

Same basic approach:

* **Vite + React + TS**
* Single page with:

  * Textarea for prompt
  * Mode toggle: “✨ Generate new” vs “🛠️ Edit existing image”
  * File upload (visible only in edit mode)
  * Settings (steps, guidance, size)
  * Image gallery of results

In the edit mode:

* You send `FormData` with `file` + `prompt`.
* Preview both original and edited image side by side.

File upload in React (simplified):

```ts
const formData = new FormData();
formData.append("file", file);        // File from <input type="file" />
formData.append("prompt", prompt);
formData.append("num_inference_steps", String(steps));
formData.append("guidance_scale", String(guidance));

const res = await fetch("http://localhost:8000/api/edit", {
  method: "POST",
  body: formData,
});
```

---

## 4. Big copy-paste prompt for your VS Code agent (with FLUX + editing)

Here’s a detailed prompt you can drop into your VS Code agent (Cline or ChatGPT Code) **from the project root**.

---

### 📌 Prompt for your coding agent

> You are my coding assistant. I’m on Ubuntu with:
>
> * Python 3.11
> * Node 22 (via nvm)
> * NVIDIA RTX 5090 (32 GB VRAM), 128 GB RAM, Threadripper 9970X
>
> I want you to build a local, GPU-accelerated image app called `flux-image-app` with:
>
> * A **backend** in Python (FastAPI) that uses Hugging Face `diffusers` FLUX pipelines:
>
>   * `FluxPipeline` for **text-to-image** using `black-forest-labs/FLUX.1-schnell`
>   * `FluxKontextPipeline` for **image editing** using `black-forest-labs/FLUX.1-Kontext-dev`
> * A **frontend** in React + TypeScript + Vite that:
>
>   * Asks the user for a **prompt**
>   * Optionally lets them **upload an image**
>   * Lets them choose between:
>
>     * “Generate new image” (text→image)
>     * “Edit uploaded image” (image + instruction, using Kontext)
>   * Shows the resulting image(s) in a gallery.
>
> ### 1. Project structure
>
> Create this structure (don’t overwrite existing files without asking):
>
> ```text
> flux-image-app/
>   backend/
>     app/
>       __init__.py
>       main.py          # FastAPI app and routes
>       schemas.py       # Pydantic models for requests/responses
>       flux_models.py   # FLUX pipeline initialization and helpers
>       config.py        # Model configuration (model ids, device, env vars)
>     requirements.txt
>     README.md
>   frontend/
>     package.json
>     tsconfig.json
>     vite.config.ts
>     index.html
>     src/
>       main.tsx
>       App.tsx
>       api.ts          # functions that call backend API
>       components/
>         PromptForm.tsx
>         ImageViewer.tsx
>         ModeToggle.tsx
>   .gitignore
>   README.md
> ```
>
> ### 2. Backend requirements and environment
>
> In `backend/requirements.txt` include at least:
>
> * `fastapi`
> * `uvicorn[standard]`
> * `pydantic`
> * `torch` (leave version as placeholder but assume CUDA build)
> * `diffusers`
> * `transformers`
> * `accelerate`
> * `safetensors`
> * `sentencepiece`
>
> In `backend/README.md` document:
>
> ```bash
> cd backend
> python -m venv .venv
> source .venv/bin/activate
> pip install --upgrade pip
> pip install -r requirements.txt
>
> # if FluxPipeline / FluxKontextPipeline are missing, also:
> # pip install -U git+https://github.com/huggingface/diffusers.git
>
> # Run the API
> uvicorn app.main:app --reload --port 8000
> ```
>
> Assume the user will set a Hugging Face token as `HUGGINGFACE_HUB_TOKEN` in their environment if FLUX models are gated.
>
> ### 3. `flux_models.py` – FLUX pipeline setup
>
> Implement `flux_models.py` so that:
>
> * It defines constants:
>
>   * `FLUX_TEXT_MODEL_ID = "black-forest-labs/FLUX.1-schnell"`
>   * `FLUX_KONTEXT_MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"`
> * It provides two lazy-loaded singletons:
>
>   * `get_text_pipeline() -> FluxPipeline`
>   * `get_kontext_pipeline() -> FluxKontextPipeline`
> * It uses `torch_dtype=torch.bfloat16` and `device="cuda"` when possible.
> * It enables performance features appropriate for a 32 GB RTX 5090:
>
>   * Do **not** CPU offload by default, but leave commented-out code showing how to enable `enable_model_cpu_offload()` if needed.
>
> Use the official diffusers usage pattern for FLUX:
>
> ```python
> from diffusers import FluxPipeline, FluxKontextPipeline
> import torch
>
> pipe = FluxPipeline.from_pretrained(
>     "black-forest-labs/FLUX.1-schnell",
>     torch_dtype=torch.bfloat16,
> )
> pipe.to("cuda")
>
> kontext = FluxKontextPipeline.from_pretrained(
>     "black-forest-labs/FLUX.1-Kontext-dev",
>     torch_dtype=torch.bfloat16,
> ).to("cuda")
> ```
>
> Make sure the module handles initialization errors cleanly (e.g., log and raise a descriptive error).
>
> ### 4. `schemas.py` – API models
>
> In `schemas.py` define:
>
> ```python
> from pydantic import BaseModel
>
> class TextGenerateRequest(BaseModel):
>     prompt: str
>     num_inference_steps: int = 4
>     guidance_scale: float = 0.0
>     width: int = 1024
>     height: int = 1024
>     seed: int | None = None
>
> class ImageResponse(BaseModel):
>     image_base64: str
> ```
>
> For editing, we’ll use multipart form for the file, so we can reuse `ImageResponse` as the response model.
>
> ### 5. `main.py` – FastAPI app and routes
>
> Implement `app/main.py` to:
>
> 1. Create a FastAPI app titled “FLUX Image API”.
> 2. Add CORS middleware allowing:
>
>    * `http://localhost:5173` and `http://127.0.0.1:5173`
> 3. Define endpoints:
>
> **Health check**
>
> ```python
> @app.get("/health")
> def health():
>     return {"status": "ok"}
> ```
>
> **Text-to-image** – `POST /api/generate`
>
> * Accept a `TextGenerateRequest` JSON body.
> * Call `get_text_pipeline()` from `flux_models.py`.
> * If a seed is provided, create a `torch.Generator("cuda")` with that seed.
> * Run the pipeline with the given parameters.
> * Convert the resulting PIL image to PNG bytes, then to base64, and return `ImageResponse(image_base64=...)`.
>
> **Image edit** – `POST /api/edit`
>
> * Accept `multipart/form-data` with fields:
>
>   * `file`: an image file (PNG/JPEG)
>   * `prompt`: text instruction
>   * optional fields: `num_inference_steps`, `guidance_scale`, `seed`
> * Use `FluxKontextPipeline` from `get_kontext_pipeline()`.
> * Load the uploaded image into a PIL Image (e.g., via `PIL.Image.open`).
> * Apply the pipeline: `pipe(image=init_image, prompt=prompt, ...)`.
> * Return `ImageResponse` with base64 PNG.
>
> Implement proper error handling and return `HTTPException` with status 400/500 when inputs are invalid or generation fails.
>
> ### 6. Frontend setup with Vite + React + TS
>
> In the `frontend` folder:
>
> 1. Initialize a Vite React + TS project:
>
> ```bash
> cd frontend
> npm init vite@latest . -- --template react-ts
> npm install
> ```
>
> 2. Configure `vite.config.ts` so the dev server runs on port 5173 by default.
>
> ### 7. Frontend API helper (`src/api.ts`)
>
> Implement an `api.ts` that exports:
>
> ```ts
> export interface TextGenerateRequest {
>   prompt: string;
>   num_inference_steps?: number;
>   guidance_scale?: number;
>   width?: number;
>   height?: number;
>   seed?: number | null;
> }
>
> export async function generateImage(req: TextGenerateRequest): Promise<string>;
> export async function editImage(
>   file: File,
>   prompt: string,
>   options?: { num_inference_steps?: number; guidance_scale?: number; seed?: number | null }
> ): Promise<string>;
> ```
>
> * `generateImage` should POST JSON to `http://localhost:8000/api/generate` and return the base64 string.
> * `editImage` should POST `FormData` to `http://localhost:8000/api/edit` and return the base64 string.
>
> ### 8. `App.tsx` – UI with prompt + upload + mode toggle
>
> Implement `App.tsx` to:
>
> * Have local state for:
>
>   * `mode`: `"generate"` or `"edit"`
>   * `prompt`
>   * `steps`, `guidance`, optional `seed`
>   * `file` (uploaded File | null)
>   * `images`: array of `{ src: string; type: "generate" | "edit" }`
>   * `isLoading`, `error`
> * Render:
>
>   * A mode toggle (buttons or tabs) for Generate vs Edit.
>   * A textarea for prompt.
>   * Numerical inputs for steps and guidance.
>   * When in edit mode, a file input (`<input type="file" accept="image/*" />`) plus a preview of the selected image.
>   * A “Generate” button that:
>
>     * Calls `generateImage()` when in generate mode.
>     * Calls `editImage()` when in edit mode and a file is present.
>   * A responsive image grid that displays each resulting image:
>
>     * `src="data:image/png;base64,..."`.
>
> Handle basic validation (e.g., don’t call edit without a file) and show a simple error message if the backend returns a bad status.
>
> ### 9. Developer experience
>
> Add to the root `README.md`:
>
> * How to run backend and frontend together:
>
> ```bash
> # Terminal 1
> cd backend
> source .venv/bin/activate
> uvicorn app.main:app --reload --port 8000
>
> # Terminal 2
> cd frontend
> npm run dev
> ```
>
> * Default URLs:
>
>   * Backend: `http://localhost:8000`
>   * Frontend: `http://localhost:5173`
>
> ### 10. Code quality
>
> * Add type hints in Python everywhere.
> * Keep functions small and focused.
> * Write clear docstrings for the main functions in `flux_models.py`.
> * In React, use functional components and hooks, no class components.
>
> Please implement all of this step by step, creating the files and filling them with working starter code. After that, summarize how to run the app and how the text→image vs image edit flow works.
>
> ---
>
> End of prompt.
>
> ---

If you paste that into your VS Code agent from the project root, it should scaffold a FLUX-based app that:

* Takes a **prompt** and spits out a FLUX image using your 5090.
* Lets you **upload an image + edit prompt** and get an edited image back, all on your local GPU.

If you want, next step I can help you tighten the **prompting style specifically for FLUX** (e.g., shorter prompts, guidance=0, 1024×1024 defaults, etc.) and add “favorite prompt presets” to the UI.
