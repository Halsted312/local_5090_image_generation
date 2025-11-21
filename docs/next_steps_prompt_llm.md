Got it—that makes total sense. We’ll keep the prank architecture, but **swap out vector similarity for a small local LLM** that answers:

> “Does this new prompt mean the same thing as any of these trap prompts?”

If **yes → return the uploaded prank image**.
If **no → call FLUX as usual**.

Below is everything you can hand to your VS Code/Codex agent:

* recommended **8B local model**
* backend stubs (LLM matcher, DB, endpoints)
* SQL migration
* storage on disk
* how this plugs into your existing FLUX API & React layout

I’ll assume the current codebase is what you uploaded: FastAPI+FLUX backend and React frontend.      

---

## 1. Choose the local LLM (8B, English, HF)

Use **Meta Llama 3 / 3.1 family, 8B instruct**:

* **`meta-llama/Meta-Llama-3-8B-Instruct`** – 8B instruct-tuned, optimized for dialogue / instruction-following, good general English reasoning, and officially supported in `transformers`. ([Hugging Face][1])
* Later you can swap to `meta-llama/Llama-3.1-8B-Instruct` or a Llama-3.2-8B variant if you want; 3.x 8B instruct models are widely regarded as among the best sub-10B open models in 2025. ([Hugging Face][2])

### Install + setup

Tell the agent to:

```bash
pip install "transformers>=4.45.0" "accelerate>=0.33.0" safetensors

# If you haven't already:
pip install psycopg2-binary sqlalchemy

# For local disk prank images:
pip install pillow
```

Login to Hugging Face and accept the Llama 3 license:

```bash
huggingface-cli login
# then accept the Meta Llama 3 license in the HF UI for the chosen model
```

Set an env var for the prank model:

```bash
export PRANK_LLM_ID="meta-llama/Meta-Llama-3-8B-Instruct"
# and your HF token (same as you use for FLUX):
export HUGGINGFACE_HUB_TOKEN="hf_..."
```

---

## 2. Database schema (Postgres) – prompts + images only

We **don’t** store embeddings anymore—only text prompts and file paths.

### SQL migration (Postgres)

Create a migration like:

```sql
-- Enable UUIDs if not already
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

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

CREATE INDEX prank_slug_idx ON pranks(slug);
CREATE INDEX prank_triggers_prank_id_idx ON prank_triggers(prank_id);
```

You can run this with Alembic or manually.

---

## 3. Backend: DB + storage + LLM matcher

Your current backend already exposes `/api/generate` and `/api/edit` via FLUX.  

We’ll **add new modules** to the same package (where `main.py` and `flux_models.py` live):

### 3.1 `database.py`

```python
# backend/database.py
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://imgen_user:imgen_pass@localhost/imgen_db",
)

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

### 3.2 `models.py`

```python
# backend/models.py
from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class Prank(Base):
    __tablename__ = "pranks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    slug = Column(String(16), unique=True, index=True, nullable=False)
    title = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    triggers = relationship(
        "PrankTrigger",
        back_populates="prank",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class PrankTrigger(Base):
    __tablename__ = "prank_triggers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prank_id = Column(UUID(as_uuid=True), ForeignKey("pranks.id", ondelete="CASCADE"), index=True, nullable=False)
    trigger_text = Column(Text, nullable=False)
    image_path = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    prank = relationship("Prank", back_populates="triggers")
```

---

### 3.3 `storage.py` (save prank images on disk)

```python
# backend/storage.py
from __future__ import annotations

import base64
import os
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PRANK_IMAGE_ROOT = Path(os.getenv("PRANK_IMAGE_ROOT", BASE_DIR / "prank_images"))
PRANK_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)


def save_prank_image(slug: str, payload: bytes, extension: str = ".png") -> str:
    """
    Save prank image bytes under ./prank_images/<slug>/<uuid>.ext and
    return the absolute path as a string.
    """
    safe_slug = slug.replace("/", "_")
    folder = PRANK_IMAGE_ROOT / safe_slug
    folder.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}{extension}"
    path = folder / filename
    with open(path, "wb") as f:
        f.write(payload)
    return str(path)


def load_prank_image_base64(path: str) -> str:
    """
    Read image bytes and return a base64-encoded PNG/bytes string.
    """
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")
```

Storing images on disk + paths in Postgres is the usual pattern for image-heavy apps; you can later swap the folder to an S3 bucket without touching DB schema. ([n8n Blog][3])

---

### 3.4 `llm_matcher.py` – **LLM-based “same prompt?” classifier**

This is the key change: instead of cosine similarity, you call Llama 3 and ask:

> *“Does USER_PROMPT mean the same thing as any of these trap prompts? Answer JSON only.”*

```python
# backend/llm_matcher.py
from __future__ import annotations

import json
import logging
import os
import threading
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # override via PRANK_LLM_ID
_MODEL_ID = os.getenv("PRANK_LLM_ID", DEFAULT_MODEL_ID)

_matcher_pipeline = None
_lock = threading.Lock()


def _get_matcher_pipeline():
    global _matcher_pipeline
    if _matcher_pipeline is None:
        with _lock:
            if _matcher_pipeline is None:
                logger.info("Loading prank LLM matcher: %s", _MODEL_ID)
                # Standard transformers text-generation pipeline; Llama 3 works well here. :contentReference[oaicite:11]{index=11}
                _matcher_pipeline = pipeline(
                    "text-generation",
                    model=_MODEL_ID,
                    tokenizer=_MODEL_ID,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
    return _matcher_pipeline


_SYSTEM_INSTRUCTIONS = (
    "You are a strict JSON-only classifier.\n"
    "Given a user prompt and a numbered list of trap prompts, decide if the user prompt "
    "has essentially the SAME intent as one of the trap prompts.\n"
    "- Ignore capitalization, punctuation, minor rewordings, and synonyms.\n"
    "- Only match if a human would say they are asking for the same image.\n"
    "- If none match, answer match=false.\n"
    "Return ONLY a JSON object, nothing else, with this exact shape:\n"
    '{"match": true|false, "index": integer or null}\n'
    "Where index is 1-based index of the matching trap prompt, or null.\n"
)


def _build_prompt(user_prompt: str, trap_prompts: List[str]) -> str:
    traps_block = "\n".join(f"{i+1}. {t}" for i, t in enumerate(trap_prompts))
    return (
        f"{_SYSTEM_INSTRUCTIONS}\n\n"
        f"TRAP_PROMPTS:\n{traps_block}\n\n"
        f"USER_PROMPT:\n{user_prompt}\n\n"
        "Now respond with the JSON object."
    )


def _extract_json(text: str) -> Optional[dict]:
    """
    Very defensive JSON extractor: grab first {...} block and parse.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def choose_matching_trigger(user_prompt: str, trap_prompts: List[str]) -> Optional[int]:
    """
    Returns the 0-based index of the matching trap prompt, or None.

    We call an 8B LLM and ask it to judge semantic equality between
    the user prompt and a small list (<= ~10) of stored prompts.
    """
    if not trap_prompts:
        return None

    pipe = _get_matcher_pipeline()
    prompt = _build_prompt(user_prompt, trap_prompts)

    # Keep it tight: short max_new_tokens, low temperature.
    outputs = pipe(
        prompt,
        max_new_tokens=64,
        do_sample=False,
        temperature=0.1,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )
    text = outputs[0]["generated_text"]
    data = _extract_json(text)
    if not data:
        logger.warning("Matcher LLM returned non-JSON: %r", text[:200])
        return None

    if not isinstance(data, dict) or not data.get("match"):
        return None

    idx = data.get("index")
    try:
        idx_int = int(idx)
    except (TypeError, ValueError):
        return None

    # LLM is instructed to be 1-based. Convert to 0-based.
    if 1 <= idx_int <= len(trap_prompts):
        return idx_int - 1
    return None
```

Notes:

* This is **pure LLM classification**, no embeddings, no thresholds.
* Max triggers per prank: I’d cap at ~10 to keep prompts small and fast.

---

### 3.5 Extend `schemas.py`

Your existing schemas define `TextGenerateRequest` and `ImageResponse` for `/api/generate` and `/api/edit`. 

Add prank schemas:

```python
# backend/schemas.py (append to existing)
from pydantic import BaseModel, Field


class PrankMetadataCreate(BaseModel):
    title: str | None = Field(None, description="Optional title for your prank link")


class PrankCreateResponse(BaseModel):
    prank_id: str
    slug: str
    share_url: str


class PrankTriggerCreateResponse(BaseModel):
    id: str
    trigger_text: str


class PrankGenerateRequest(BaseModel):
    prompt: str = Field(..., description="User prompt on a prank page")
```

---

## 4. Wire everything into `main.py`

Your current `main.py` has the FLUX endpoints and CORS. 

We’ll:

1. Import DB + models + storage + LLM matcher.
2. Create tables at startup (for dev).
3. Add three endpoints:

   * `POST /api/pranks` – create prank (generates slug)
   * `POST /api/pranks/{prank_id}/triggers` – add (prompt, image) rows
   * `POST /api/p/{slug}/generate` – core “trick” endpoint

### 4.1. Update imports at top of `main.py`

Add to existing imports:

```python
# main.py
from typing import Iterable

import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .flux_models import get_kontext_pipeline, get_text_pipeline
from .schemas import (
    ImageResponse,
    TextGenerateRequest,
    PrankMetadataCreate,
    PrankCreateResponse,
    PrankTriggerCreateResponse,
    PrankGenerateRequest,
)
from .database import Base, engine, get_db
from .models import Prank, PrankTrigger
from .storage import save_prank_image, load_prank_image_base64
from .llm_matcher import choose_matching_trigger
```

(Keep your existing imports that were already there.)

### 4.2. Create tables at startup

Right after `app = FastAPI(...)` add:

```python
Base.metadata.create_all(bind=engine)
```

### 4.3. Slug generator helper

Add near the top of the file:

```python
import random
import string

SLUG_ALPHABET = string.ascii_letters + string.digits


def _generate_unique_slug(db, length: int = 8) -> str:
    while True:
        candidate = "".join(random.choices(SLUG_ALPHABET, k=length))
        existing = db.query(Prank).filter(Prank.slug == candidate).first()
        if existing is None:
            return candidate
```

---

### 4.4. Endpoint: create prank metadata

```python
@app.post("/api/pranks", response_model=PrankCreateResponse)
def create_prank(
    metadata: PrankMetadataCreate,
    db=Depends(get_db),
) -> PrankCreateResponse:
    slug = _generate_unique_slug(db, length=8)
    prank = Prank(
        slug=slug,
        title=metadata.title,
    )
    db.add(prank)
    db.commit()
    db.refresh(prank)

    share_url = f"/p/{prank.slug}"
    return PrankCreateResponse(
        prank_id=str(prank.id),
        slug=prank.slug,
        share_url=share_url,
    )
```

---

### 4.5. Endpoint: add trigger (prompt + uploaded image)

```python
@app.post("/api/pranks/{prank_id}/triggers", response_model=PrankTriggerCreateResponse)
async def add_prank_trigger(
    prank_id: str,
    trigger_text: str = Form(...),
    file: UploadFile = File(...),
    db=Depends(get_db),
) -> PrankTriggerCreateResponse:
    prank = db.query(Prank).filter(Prank.id == prank_id).first()
    if prank is None:
        raise HTTPException(status_code=404, detail="Prank not found")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    payload = await file.read()

    extension = ".png"
    if file.filename and "." in file.filename:
        extension = "." + file.filename.rsplit(".", 1)[-1].lower()

    image_path = save_prank_image(prank.slug, payload, extension=extension)

    trigger = PrankTrigger(
        prank_id=prank.id,
        trigger_text=trigger_text,
        image_path=image_path,
    )
    db.add(trigger)
    db.commit()
    db.refresh(trigger)

    return PrankTriggerCreateResponse(id=str(trigger.id), trigger_text=trigger.trigger_text)
```

---

### 4.6. Endpoint: prank page generation `/api/p/{slug}/generate`

This is where the **LLM decides** whether to show prank image or call FLUX.

```python
@app.post("/api/p/{slug}/generate", response_model=ImageResponse)
def generate_prank_image(
    slug: str,
    request: PrankGenerateRequest,
    db=Depends(get_db),
) -> ImageResponse:
    prank = db.query(Prank).filter(Prank.slug == slug).first()
    if prank is None:
        raise HTTPException(status_code=404, detail="Prank not found")

    triggers = db.query(PrankTrigger).filter(PrankTrigger.prank_id == prank.id).all()

    # If no triggers exist, just behave like the normal generator.
    if not triggers:
        base_request = TextGenerateRequest(
            prompt=request.prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
            width=1024,
            height=1024,
            seed=None,
        )
        return generate_image(base_request)

    trap_texts = [t.trigger_text for t in triggers]
    idx = choose_matching_trigger(request.prompt, trap_texts)

    if idx is not None:
        # LLM says this matches one of the trap prompts -> show prank image
        trigger = triggers[idx]
        image_base64 = load_prank_image_base64(trigger.image_path)
        return ImageResponse(image_base64=image_base64)

    # Otherwise, call normal FLUX text-to-image pipeline
    base_request = TextGenerateRequest(
        prompt=request.prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        width=1024,
        height=1024,
        seed=None,
    )
    return generate_image(base_request)
```

This reuses your existing FLUX generation logic in `generate_image`. 

---

## 5. Frontend layout: how it ties together

Right now your frontend has:

* `App.tsx` – main UI with generate/edit + sliders, reusing `PromptForm` and `ImageViewer`.   
* `main.tsx` – mounts `<App />` into `#root`. 
* `index.css` – already mobile-friendly with a two-panel layout that stacks on small screens. 

For the trick site behavior you described:

1. **Main page** (`/`): keep as your “Imgen 4 U” normal generator (text-to-image + optional edit). Visitors here never see prank logic.
2. **Prank builder** (`/create`): UI where you:

   * Enter a title (optional).
   * Add multiple rows: “trigger prompt” + “upload image”.
   * Hit “Create shareable link” → calls `POST /api/pranks`, then `POST /api/pranks/{id}/triggers` for each row.
   * Shows resulting share URL `/p/<slug>`.
3. **Prank page** (`/p/:slug`): looks almost identical to main generator layout, but:

   * On submit, calls `POST /api/p/{slug}/generate`.
   * Backend decides prank vs real FLUX.

You already have good CSS and panel grid, so:

* Reuse that structure for **PrankCreateApp** and **PrankPlayApp**.
* For mobile: `.layout` already collapses to 1 column at `<900px`. 
* Use the right panel for output, left for the form—your current layout already does that.

In the previous message I gave you full TSX stubs for `PrankCreateApp` and `PrankPlayApp`. With this new LLM matcher, **you don’t need to change any frontend code**—only the backend logic changed from embeddings → LLM classification. The only frontend requirement is:

* `PrankPlayApp` uses `fetch("/api/p/" + slug + "/generate")` instead of `/api/generate`.

Everything else is identical from the browser’s point of view.

---

## 6. What to literally tell your VS Code agent

You can paste something like this into Codex/Cloud Agent:

1. **Add dependencies:**

   * `transformers`, `accelerate`, `safetensors`, `sqlalchemy`, `psycopg2-binary`, `pillow`.

2. **Backend changes:**

   * Create `database.py` and `models.py` exactly as above.
   * Create `storage.py` to save prank images locally under `./prank_images/<slug>/...`.
   * Create `llm_matcher.py` that loads `meta-llama/Meta-Llama-3-8B-Instruct` (or `PRANK_LLM_ID`) using a transformers `pipeline("text-generation")`, and implements `choose_matching_trigger(user_prompt, trap_prompts)` using a JSON-only system prompt.
   * Extend `schemas.py` with `PrankMetadataCreate`, `PrankCreateResponse`, `PrankTriggerCreateResponse`, `PrankGenerateRequest`.
   * In `main.py`:

     * Import DB, models, storage, and `choose_matching_trigger`.
     * Call `Base.metadata.create_all(bind=engine)` on startup (dev only).
     * Add `_generate_unique_slug` helper.
     * Implement:

       * `POST /api/pranks` → create prank, return `{prank_id, slug, share_url}`.
       * `POST /api/pranks/{prank_id}/triggers` → upload image + trigger_text, save via `save_prank_image`, persist in `prank_triggers`.
       * `POST /api/p/{slug}/generate` → lookup prank + triggers, call `choose_matching_trigger`, then either:

         * return prank image via `load_prank_image_base64`, or
         * call existing `generate_image` (FLUX) with the user prompt and return that image.

3. **Frontend:**

   * Keep your existing `App.tsx`, `PromptForm`, `ImageViewer`, and layout.
   * Add `/create` + `/p/:slug` pages as previously discussed; those just change which endpoint they call (normal vs prank). No awareness of LLM vs similarity is needed on the client.

