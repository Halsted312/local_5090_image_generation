Here’s a concrete plan + stub code you can hand straight to your coding agent.

> **Note:** The current direction is text-to-image only—no user uploads. Keep the FLUX prank concepts for inspiration, but skip the upload-backed steps when applying this outline.

I’ll:

* Plug a **local “small model”** for prompt similarity on your 5090 (BGE small via `sentence-transformers`) ([Hugging Face][1])
* Add **Postgres tables + SQL migration** for pranks + triggers
* Add **disk storage** for prank images on your machine ([Stack Overflow][2])
* Wire it all into your existing **FastAPI + FLUX** backend
* Restructure the React layout into:

  * `/` – normal generator (no upload)
  * `/create` – prank builder with upload
  * `/p/:slug` – prank page that looks like a normal generator

---

## 1. Backend: DB, similarity model, and storage

### 1.1 SQL migration (Postgres)

Create a migration file (or run manually) like:

```sql
-- Enable UUIDs (if not already)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Prank definitions
CREATE TABLE pranks (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  slug VARCHAR(16) UNIQUE NOT NULL,
  title TEXT,
  similarity_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.88,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per "trap" prompt + image
CREATE TABLE prank_triggers (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  prank_id UUID NOT NULL REFERENCES pranks(id) ON DELETE CASCADE,
  trigger_text TEXT NOT NULL,
  image_path TEXT NOT NULL,
  embedding DOUBLE PRECISION[],         -- 1D array of floats (384-dim)
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX prank_slug_idx ON pranks(slug);
CREATE INDEX prank_triggers_prank_id_idx ON prank_triggers(prank_id);
```

* `embedding` will hold a 384‑dim vector from **BAAI/bge-small-en-v1.5** (good small model for semantic similarity). ([Hugging Face][1])

---

### 1.2. New `database.py` (SQLAlchemy glue)

Create `backend/database.py`:

```python
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Set this in your env, e.g.
# export DATABASE_URL="postgresql+psycopg2://imgen_user:imgen_pass@localhost/imgen_db"
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://imgen_user:imgen_pass@localhost/imgen_db",
)

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """
    FastAPI dependency that yields a DB session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

### 1.3. New `models.py` (pranks + triggers)

Create `backend/models.py`:

```python
from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class Prank(Base):
    __tablename__ = "pranks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    slug = Column(String(16), unique=True, nullable=False, index=True)
    title = Column(Text, nullable=True)
    similarity_threshold = Column(Float, nullable=False, default=0.88)
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
    prank_id = Column(UUID(as_uuid=True), ForeignKey("pranks.id", ondelete="CASCADE"), nullable=False, index=True)
    trigger_text = Column(Text, nullable=False)
    image_path = Column(Text, nullable=False)
    # 384-dim embedding from BAAI/bge-small-en-v1.5
    embedding = Column(ARRAY(Float), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    prank = relationship("Prank", back_populates="triggers")
```

---

### 1.4. New `similarity_model.py` (your “small model” on 5090)

See the next_steps_prompt_llm.md in /docs folder

This model runs locally on your GPU and decides whether a friend’s prompt is “close enough” to the original trap prompt.

---

### 1.5. New `storage.py` (store prank images on disk)

Store prank images under a local directory like `./prank_images/<slug>/...`. Best practice is to keep binary files in the filesystem or object storage and only store the path/URL in Postgres. ([Stack Overflow][2])

Create `backend/storage.py`:

```python
from __future__ import annotations

import base64
import logging
import os
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# Project root (adjust if your package layout differs)
BASE_DIR = Path(__file__).resolve().parent.parent
PRANK_IMAGE_ROOT = Path(os.getenv("PRANK_IMAGE_ROOT", BASE_DIR / "prank_images"))

PRANK_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)


def save_prank_image(slug: str, payload: bytes, extension: str = ".png") -> str:
    """
    Save prank image bytes in a slug-specific folder.
    Returns the absolute path as a string.
    """
    safe_slug = slug.replace("/", "_")
    folder = PRANK_IMAGE_ROOT / safe_slug
    folder.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}{extension}"
    path = folder / filename

    logger.info("Saving prank image for slug %s to %s", slug, path)
    with open(path, "wb") as f:
        f.write(payload)

    return str(path)


def load_prank_image_base64(path: str) -> str:
    """
    Read an image from disk and return it as base64 string (PNG or original).
    """
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")
```

---

### 1.6. Extend `schemas.py` (FastAPI models)

You already have `TextGenerateRequest` and `ImageResponse`. 
Add new models:

```python
from pydantic import BaseModel, Field

# ... existing TextGenerateRequest and ImageResponse ...


class PrankMetadataCreate(BaseModel):
  title: str | None = Field(None, description="Optional title for your prank link")
  similarity_threshold: float = Field(
      0.88, ge=0.0, le=1.0, description="Cosine similarity threshold for trap activation"
  )


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

### 1.7. Update `main.py` (add prank endpoints)

You already have the FLUX API for `/api/generate` and `/api/edit`.

Now extend it.

**Imports at top:**

```python
from typing import Iterable

import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import Prank, PrankTrigger
from .similarity_model import cosine_similarity, embed_text
from .storage import load_prank_image_base64, save_prank_image
from .schemas import (
    ImageResponse,
    PrankCreateResponse,
    PrankGenerateRequest,
    PrankMetadataCreate,
    PrankTriggerCreateResponse,
    TextGenerateRequest,
)
```

Right after you create `app = FastAPI(...)`, run:

```python
# Ensure tables exist in local dev (in production you'd use migrations)
Base.metadata.create_all(bind=engine)
```

#### 1.7.1 Helper to generate random slugs

Add near top-level:

```python
import random
import string

SLUG_ALPHABET = string.ascii_letters + string.digits


def _generate_unique_slug(db: Session, length: int = 8) -> str:
    while True:
        candidate = "".join(random.choices(SLUG_ALPHABET, k=length))
        existing = db.query(Prank).filter(Prank.slug == candidate).first()
        if existing is None:
            return candidate
```

#### 1.7.2 `POST /api/pranks` – create prank metadata

```python
@app.post("/api/pranks", response_model=PrankCreateResponse)
def create_prank(
    metadata: PrankMetadataCreate,
    db: Session = Depends(get_db),
) -> PrankCreateResponse:
    slug = _generate_unique_slug(db, length=8)
    prank = Prank(
        slug=slug,
        title=metadata.title,
        similarity_threshold=metadata.similarity_threshold,
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

#### 1.7.3 `POST /api/pranks/{prank_id}/triggers` – add trigger rows

```python
@app.post("/api/pranks/{prank_id}/triggers", response_model=PrankTriggerCreateResponse)
async def add_prank_trigger(
    prank_id: str,
    trigger_text: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> PrankTriggerCreateResponse:
    prank = db.query(Prank).filter(Prank.id == prank_id).first()
    if prank is None:
        raise HTTPException(status_code=404, detail="Prank not found")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    payload = await file.read()

    # Decide on an extension
    ext = ".png"
    if file.filename and "." in file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower()

    image_path = save_prank_image(prank.slug, payload, extension=ext)

    embedding = embed_text(trigger_text)

    trigger = PrankTrigger(
        prank_id=prank.id,
        trigger_text=trigger_text,
        image_path=image_path,
        embedding=embedding,
    )
    db.add(trigger)
    db.commit()
    db.refresh(trigger)

    return PrankTriggerCreateResponse(id=str(trigger.id), trigger_text=trigger.trigger_text)
```

#### 1.7.4 `POST /api/p/{slug}/generate` – prank vs real image

```python
@app.post("/api/p/{slug}/generate", response_model=ImageResponse)
def generate_prank_image(
    slug: str,
    request: PrankGenerateRequest,
    db: Session = Depends(get_db),
) -> ImageResponse:
    prank = db.query(Prank).filter(Prank.slug == slug).first()
    if prank is None:
        raise HTTPException(status_code=404, detail="Prank not found")

    triggers = db.query(PrankTrigger).filter(PrankTrigger.prank_id == prank.id).all()
    if not triggers:
        # Fallback: no triggers, just behave like normal generator
        base_request = TextGenerateRequest(
            prompt=request.prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
            width=1024,
            height=1024,
            seed=None,
        )
        return generate_image(base_request)

    prompt_embedding = embed_text(request.prompt)

    best_trigger = None
    best_score = 0.0
    for trigger in triggers:
        if trigger.embedding is None:
            continue
        score = cosine_similarity(prompt_embedding, trigger.embedding)
        if score > best_score:
            best_score = score
            best_trigger = trigger

    if best_trigger is not None and best_score >= prank.similarity_threshold:
        # Return the prank image stored on disk as base64
        image_base64 = load_prank_image_base64(best_trigger.image_path)
        return ImageResponse(image_base64=image_base64)

    # Otherwise, call the normal FLUX text-to-image pipeline
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

That’s the “fake vs real” decision logic wired into your existing FLUX generator.

---

## 2. Frontend: layout + new pages

Right now you have a single `App` that handles generate + edit with a mode toggle, and a single entry in `main.tsx`.

We’ll split into:

* `App.tsx` – main generator (`/`) – **no upload, just text-to-image**
* `PrankCreateApp.tsx` – prank builder (`/create`)
* `PrankPlayApp.tsx` – prank page (`/p/:slug`)

And update `main.tsx` to route based on `window.location.pathname`.

---

### 2.1. Update `main.tsx` – simple path-based router

Replace `main.tsx` with:

```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import PrankCreateApp from "./PrankCreateApp";
import PrankPlayApp from "./PrankPlayApp";
import "./index.css";

const rootElement = document.getElementById("root") as HTMLElement;
const root = ReactDOM.createRoot(rootElement);

const path = window.location.pathname;

let ui: React.ReactElement;

if (path === "/create") {
  ui = <PrankCreateApp />;
} else if (path.startsWith("/p/")) {
  const [, , slug] = path.split("/");
  ui = <PrankPlayApp slug={slug ?? ""} />;
} else {
  // default landing: normal generator
  ui = <App />;
}

root.render(
  <React.StrictMode>
    {ui}
  </React.StrictMode>,
);
```

No extra dependencies (like React Router) needed.

---

### 2.2. Replace `App.tsx` – clean main generator (no upload)

Simplify `App.tsx` to only generate from text using your existing FLUX `/api/generate` endpoint.

```tsx
import { useMemo, useState } from "react";
import PromptForm from "./components/PromptForm";
import ImageViewer, { GeneratedImage } from "./components/ImageViewer";
import { generateImage } from "./api";

type Mode = "generate";

function makeId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2);
}

export default function App() {
  const [prompt, setPrompt] = useState("show me a cherry tree on a hill");
  const [steps, setSteps] = useState(6);
  const [guidance, setGuidance] = useState(2);
  const [images, setImages] = useState<GeneratedImage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const statusText = useMemo(() => {
    if (isLoading) return "Working with FLUX...";
    if (error) return null;
    if (images.length > 0) return "Done. Try another idea.";
    return null;
  }, [error, images.length, isLoading]);

  const handleSubmit = async () => {
    setError(null);
    const trimmed = prompt.trim();
    if (!trimmed) {
      setError("Prompt is required.");
      return;
    }

    setIsLoading(true);
    try {
      // Map UI sliders into FLUX params
      const mappedSteps = Math.round(4 + (steps - 1) * 0.5); // ~4–8.5
      const mappedGuidance = guidance === 1 ? 0 : guidance - 1; // 0–2

      const imageBase64 = await generateImage({
        prompt: trimmed,
        num_inference_steps: mappedSteps,
        guidance_scale: mappedGuidance,
        width: 1024,
        height: 1024,
      });

      const newImage: GeneratedImage = {
        id: makeId(),
        src: imageBase64,
        mode: "generate",
        prompt: trimmed,
      };
      setImages((prev) => [newImage, ...prev]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="page">
      <header className="header">
        <div>
          <div className="title">Imgen 4 U</div>
          <p style={{ color: "#9ca3af", marginTop: "0.2rem" }}>
            Describe the image you want – FLUX will render it.
          </p>
        </div>
        <a className="pill" href="/create">
          Create a prank link →
        </a>
      </header>

      <div className="layout">
        <section className="panel">
          <div className="section-title">Describe your image</div>
          <PromptForm
            mode="generate"
            prompt={prompt}
            steps={steps}
            guidance={guidance}
            file={null}
            error={error}
            isLoading={isLoading}
            onPromptChange={setPrompt}
            onStepsChange={setSteps}
            onGuidanceChange={setGuidance}
            onFileChange={() => {
              // no-op; upload not used on main page
            }}
            onSubmit={handleSubmit}
          />
          {statusText && (
            <div className="status" style={{ marginTop: "0.5rem" }}>
              {statusText}
            </div>
          )}
        </section>

        <section className="panel">
          <div className="section-title">Output</div>
          <ImageViewer images={images} />
        </section>
      </div>
    </div>
  );
}
```

This keeps your nice sliders + prompt layout, but removes the image upload mode from the main site.

(`PromptForm` already hides the upload controls when `mode === "generate"` .)

---

### 2.3. New `PrankCreateApp.tsx` – prank builder at `/create`

Create `PrankCreateApp.tsx` next to `App.tsx`:

```tsx
import { useEffect, useMemo, useState } from "react";
import "./index.css";
import { createPrank, addPrankTrigger } from "./api";

interface TriggerRow {
  id: string;
  triggerText: string;
  file: File | null;
  previewUrl?: string;
}

function makeId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2);
}

export default function PrankCreateApp() {
  const [title, setTitle] = useState("");
  const [threshold, setThreshold] = useState(0.88);
  const [rows, setRows] = useState<TriggerRow[]>([
    { id: makeId(), triggerText: "", file: null, previewUrl: undefined },
  ]);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shareUrl, setShareUrl] = useState<string | null>(null);

  const validRows = useMemo(
    () => rows.filter((r) => r.triggerText.trim() && r.file),
    [rows],
  );

  const handleAddRow = () => {
    setRows((prev) => [...prev, { id: makeId(), triggerText: "", file: null }]);
  };

  const handleRemoveRow = (id: string) => {
    setRows((prev) => prev.filter((r) => r.id !== id));
  };

  const handleTextChange = (id: string, value: string) => {
    setRows((prev) => prev.map((r) => (r.id === id ? { ...r, triggerText: value } : r)));
  };

  const handleFileChange = (id: string, file: File | null) => {
    setRows((prev) =>
      prev.map((r) => {
        if (r.id !== id) return r;
        if (r.previewUrl) {
          URL.revokeObjectURL(r.previewUrl);
        }
        return {
          ...r,
          file,
          previewUrl: file ? URL.createObjectURL(file) : undefined,
        };
      }),
    );
  };

  useEffect(
    () => () => {
      // Clean up any preview URLs on unmount
      rows.forEach((r) => {
        if (r.previewUrl) URL.revokeObjectURL(r.previewUrl);
      });
    },
    [rows],
  );

  const handleCreate = async () => {
    setError(null);

    if (validRows.length === 0) {
      setError("Add at least one trigger prompt + image.");
      return;
    }

    setIsSaving(true);
    try {
      const prank = await createPrank({
        title: title.trim() || undefined,
        similarity_threshold: threshold,
      });

      for (const row of validRows) {
        await addPrankTrigger(prank.prank_id, row.triggerText, row.file!);
      }

      const fullUrl = `${window.location.origin}/p/${prank.slug}`;
      setShareUrl(fullUrl);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create prank");
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="page">
      <header className="header">
        <div>
          <div className="title">Imgen 4 U — Prank Builder</div>
          <p style={{ color: "#9ca3af", marginTop: "0.2rem" }}>
            Create a secret link that behaves like a normal generator, except for the prompts you choose.
          </p>
        </div>
        <a className="pill" href="/">
          ← Back to generator
        </a>
      </header>

      <main className="panel">
        <div className="section-title">1. Prank settings</div>
        <div className="field">
          <label htmlFor="title">Title (optional)</label>
          <input
            id="title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Alex's prank link"
          />
        </div>

        <div className="field" style={{ marginTop: "0.75rem" }}>
          <label htmlFor="threshold">
            Matching strictness ({threshold.toFixed(2)})
          </label>
          <small className="hint">
            0.8 = loose, 0.9+ = only very similar prompts trigger the prank image.
          </small>
          <input
            id="threshold"
            type="range"
            min={0.7}
            max={0.98}
            step={0.01}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
          />
        </div>

        <div className="section-title" style={{ marginTop: "1.5rem" }}>
          2. Triggers
        </div>
        <p style={{ color: "#9ca3af", fontSize: "0.9rem", marginBottom: "0.75rem" }}>
          For each row, write the kind of prompt your friend might type and upload the image you want to show instead.
        </p>

        <div className="trigger-list">
          {rows.map((row) => (
            <div className="trigger-row" key={row.id}>
              <div className="field">
                <label>Trigger prompt</label>
                <textarea
                  value={row.triggerText}
                  onChange={(e) => handleTextChange(row.id, e.target.value)}
                  placeholder='e.g. "who is the most beautiful girl in the world?"'
                />
              </div>
              <div className="field">
                <label>Prank image</label>
                <label className="upload-cta inline">
                  <span>{row.file ? "Change image" : "Choose image"}</span>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => handleFileChange(row.id, e.target.files?.[0] ?? null)}
                  />
                </label>
                {row.file && (
                  <div className="file-info">
                    Selected: {row.file.name}
                  </div>
                )}
                {row.previewUrl && (
                  <div className="file-preview">
                    <img src={row.previewUrl} alt="Preview" />
                  </div>
                )}
              </div>
              <button
                type="button"
                className="button"
                style={{ marginTop: "0.75rem", background: "rgba(248,113,113,0.15)", color: "#fecaca" }}
                onClick={() => handleRemoveRow(row.id)}
              >
                Remove trigger
              </button>
            </div>
          ))}
        </div>

        <button
          type="button"
          className="button"
          style={{ marginTop: "1rem" }}
          onClick={handleAddRow}
        >
          + Add another trigger
        </button>

        <div className="actions-row" style={{ marginTop: "1.5rem" }}>
          <div className="action-left" />
          <div className="action-right">
            <button
              className="button"
              type="button"
              disabled={isSaving}
              onClick={handleCreate}
            >
              {isSaving ? "Creating link..." : "Create shareable link"}
            </button>
            {error && <span className="error">{error}</span>}
          </div>
        </div>

        {shareUrl && (
          <div className="glass" style={{ marginTop: "1.5rem" }}>
            <div className="section-title">3. Share</div>
            <p style={{ color: "#e5e7eb", fontSize: "0.95rem" }}>
              Send this link to your friend. It looks like a normal image generator, but your triggers will show the prank images.
            </p>
            <div className="field" style={{ marginTop: "0.75rem" }}>
              <input
                readOnly
                value={shareUrl}
                onFocus={(e) => e.target.select()}
              />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
```

This gives you that “row-by-row with thumbnail” UI you described, and it calls the backend per row.

---

### 2.4. New `PrankPlayApp.tsx` – shared prank page `/p/:slug`

Create `PrankPlayApp.tsx`:

```tsx
import { useMemo, useState } from "react";
import "./index.css";
import PromptForm from "./components/PromptForm";
import ImageViewer, { GeneratedImage } from "./components/ImageViewer";
import { generatePrankImage } from "./api";

type Mode = "generate";

interface PrankPlayAppProps {
  slug: string;
}

function makeId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2);
}

export default function PrankPlayApp({ slug }: PrankPlayAppProps) {
  const [prompt, setPrompt] = useState("");
  const [steps, setSteps] = useState(6);
  const [guidance, setGuidance] = useState(2);
  const [images, setImages] = useState<GeneratedImage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const statusText = useMemo(() => {
    if (isLoading) return "Generating image...";
    if (error) return null;
    if (images.length > 0) return "Done. Try another idea.";
    return null;
  }, [error, images.length, isLoading]);

  const handleSubmit = async () => {
    setError(null);
    const trimmed = prompt.trim();
    if (!trimmed) {
      setError("Prompt is required.");
      return;
    }

    setIsLoading(true);
    try {
      // For prank pages we can keep a simple mapping.
      const mappedSteps = Math.round(4 + (steps - 1) * 0.5);
      const mappedGuidance = guidance === 1 ? 0 : guidance - 1;

      const imageBase64 = await generatePrankImage(slug, trimmed, {
        num_inference_steps: mappedSteps,
        guidance_scale: mappedGuidance,
      });

      const newImage: GeneratedImage = {
        id: makeId(),
        src: imageBase64,
        mode: "generate",
        prompt: trimmed,
      };
      setImages((prev) => [newImage, ...prev]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="page">
      <header className="header">
        <div className="title">Imgen 4 U</div>
        <p style={{ color: "#9ca3af", marginTop: "0.2rem" }}>
          Type a description and we&apos;ll generate an image for you.
        </p>
      </header>

      <div className="layout">
        <section className="panel">
          <div className="section-title">Describe your image</div>
          <PromptForm
            mode="generate"
            prompt={prompt}
            steps={steps}
            guidance={guidance}
            file={null}
            error={error}
            isLoading={isLoading}
            onPromptChange={setPrompt}
            onStepsChange={setSteps}
            onGuidanceChange={setGuidance}
            onFileChange={() => {
              // no upload here
            }}
            onSubmit={handleSubmit}
          />
          {statusText && (
            <div className="status" style={{ marginTop: "0.5rem" }}>
              {statusText}
            </div>
          )}
        </section>

        <section className="panel">
          <div className="section-title">Output</div>
          <ImageViewer images={images} />
        </section>
      </div>
    </div>
  );
}
```

To the friend, this looks like a regular text‑to‑image page. Under the hood, it calls your new `/api/p/{slug}/generate` endpoint.

---

### 2.5. Update `api.ts` – add prank endpoints

You already have something like this file used by `App.tsx` (`generateImage`, `editImage`). 

Extend it with new functions:

```ts
// api.ts
const BASE_URL = "http://localhost:8000"; // adjust to your FastAPI host

interface GeneratePayload {
  prompt: string;
  num_inference_steps: number;
  guidance_scale: number;
  width: number;
  height: number;
}

export async function generateImage(payload: GeneratePayload): Promise<string> {
  const res = await fetch(`${BASE_URL}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error("Generation failed");
  }
  const data = await res.json();
  return data.image_base64 as string;
}

// Existing editImage(...) stays as-is for your own tool, if you still want it.

// --- New prank APIs ---

interface PrankMetadata {
  title?: string;
  similarity_threshold: number;
}

interface PrankCreateResponse {
  prank_id: string;
  slug: string;
  share_url: string;
}

export async function createPrank(metadata: PrankMetadata): Promise<PrankCreateResponse> {
  const res = await fetch(`${BASE_URL}/api/pranks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(metadata),
  });
  if (!res.ok) {
    throw new Error("Failed to create prank");
  }
  return (await res.json()) as PrankCreateResponse;
}

export async function addPrankTrigger(prankId: string, triggerText: string, file: File): Promise<void> {
  const form = new FormData();
  form.append("trigger_text", triggerText);
  form.append("file", file);

  const res = await fetch(`${BASE_URL}/api/pranks/${prankId}/triggers`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    throw new Error("Failed to add prank trigger");
  }
}

interface PrankGenerateOptions {
  num_inference_steps: number;
  guidance_scale: number;
}

export async function generatePrankImage(
  slug: string,
  prompt: string,
  opts: PrankGenerateOptions,
): Promise<string> {
  const res = await fetch(`${BASE_URL}/api/p/${slug}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) {
    throw new Error("Prank generation failed");
  }
  const data = await res.json();
  return data.image_base64 as string;
}
```

(Notice for now, `num_inference_steps` and `guidance_scale` on prank pages are not sent to the backend; you can wire them up later if you want.)

---

### 2.6. CSS tweaks for builder layout

You already have a nice glassmorphism look, responsive grid, etc. 

Add a few new rules at the bottom of `index.css`:

```css
.trigger-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-top: 0.75rem;
}

.trigger-row {
  border-radius: 16px;
  padding: 1rem;
  background: rgba(15, 23, 42, 0.7);
  border: 1px solid rgba(148, 163, 184, 0.3);
}

.trigger-row .field textarea {
  min-height: 80px;
}

@media (max-width: 900px) {
  .trigger-row {
    padding: 0.75rem;
  }
}
```

This keeps everything mobile‑friendly and fits with your existing design.

---

## 3. Summary for your coding agent

You can literally tell your agent:

1. **Backend**

   * Add `database.py`, `models.py`, `similarity_model.py`, and `storage.py` exactly as above.
   * Add the new Pydantic models to `schemas.py`.
   * Wire the DB + prank endpoints into `main.py`:

     * `POST /api/pranks`
     * `POST /api/pranks/{prank_id}/triggers`
     * `POST /api/p/{slug}/generate`
   * Use `BAAI/bge-small-en-v1.5` via `sentence-transformers` to embed texts and compute cosine similarity on GPU.
   * Store prank images on disk under `./prank_images/<slug>/...` and only persist `image_path` + embedding in Postgres.

2. **Frontend**

   * Update `main.tsx` to choose between:

     * `/` → `App`
     * `/create` → `PrankCreateApp`
     * `/p/:slug` → `PrankPlayApp`
   * Replace `App.tsx` with the simplified text-only generator.
   * Add `PrankCreateApp.tsx` for building pranks and `PrankPlayApp.tsx` for shared links.
   * Extend `api.ts` with `createPrank`, `addPrankTrigger`, `generatePrankImage`.
   * Add the small CSS snippets for `.trigger-list` and `.trigger-row`.

That’s the full skeleton: FLUX still does the image work, your local BGE model decides whether to show the uploaded prank image or a real FLUX generation, and the UI splits cleanly into main, create, and prank pages.

