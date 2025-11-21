from __future__ import annotations

import base64
import io
import logging
import os
import random
import string
from typing import Iterable

import torch
from fastapi import Depends, FastAPI, HTTPException
from fastapi import File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .flux_models import get_text_pipeline
from .llm_matcher import choose_matching_trigger
from .models import Prank, PrankTrigger
from .schemas import (
    ImageResponse,
    PrankCreateResponse,
    PrankDetailResponse,
    PrankGenerateRequest,
    PrankMetadataCreate,
    PrankTriggerCreateResponse,
    PrankTriggerInfo,
    TextGenerateRequest,
)
from .storage import (
    PRANK_IMAGE_ROOT,
    load_prank_image_base64,
    resolve_image_path,
    save_prank_image,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FLUX Image API + Prank Mode")

_env_origins = os.getenv("CORS_ALLOW_ORIGINS")
if _env_origins:
    ALLOWED_ORIGINS: Iterable[str] = tuple(o.strip() for o in _env_origins.split(",") if o.strip())
else:
    ALLOWED_ORIGINS: Iterable[str] = (
        "http://localhost:7080",
        "http://127.0.0.1:7080",
        "http://localhost:6970",
        "http://127.0.0.1:6970",
        "http://localhost",
        "http://127.0.0.1",
        "https://app.promptpics.ai",
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SLUG_ALPHABET = string.ascii_letters + string.digits


@app.on_event("startup")
def _ensure_tables() -> None:
    """Create tables on startup for local/dev usage."""
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to create tables on startup")
        raise


def _pil_to_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _make_generator(device: torch.device | str, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    return torch.Generator(device=str(device)).manual_seed(seed)


def _free_cuda_memory() -> None:
    """Best-effort VRAM cleanup after requests to reduce OOM risk."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001
            pass


def _generate_unique_slug(db: Session, length: int = 5) -> str:
    while True:
        candidate = "".join(random.choices(SLUG_ALPHABET, k=length))
        existing = db.query(Prank).filter(Prank.slug == candidate).first()
        if existing is None:
            return candidate


def _build_text_request(
    prompt: str,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
) -> TextGenerateRequest:
    return TextGenerateRequest(
        prompt=prompt,
        num_inference_steps=4 if num_inference_steps is None else num_inference_steps,
        guidance_scale=0.0 if guidance_scale is None else guidance_scale,
        width=640 if width is None else width,
        height=640 if height is None else height,
        seed=seed,
    )


def _run_flux(request: TextGenerateRequest) -> ImageResponse:
    return generate_image(request)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/generate", response_model=ImageResponse)
def generate_image(request: TextGenerateRequest) -> ImageResponse:
    pipe = get_text_pipeline()
    device = getattr(pipe, "device", "cpu")
    generator = _make_generator(device, request.seed)
    try:
        result = pipe(
            prompt=request.prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            generator=generator,
        )
        image = result.images[0]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Text-to-image generation failed")
        raise HTTPException(status_code=500, detail="Image generation failed") from exc
    finally:
        _free_cuda_memory()

    return ImageResponse(image_base64=_pil_to_base64_png(image))


@app.post("/api/pranks", response_model=PrankCreateResponse)
def create_prank(
    metadata: PrankMetadataCreate,
    db: Session = Depends(get_db),
) -> PrankCreateResponse:
    if metadata.slug:
        slug = metadata.slug.strip()
        if len(slug) < 3 or len(slug) > 16:
            raise HTTPException(status_code=400, detail="Slug must be between 3 and 16 characters.")
        existing = db.query(Prank).filter(Prank.slug == slug).first()
        if existing:
            raise HTTPException(status_code=400, detail="Slug already in use.")
    else:
        slug = _generate_unique_slug(db, length=5)
    prank = Prank(slug=slug, title=metadata.title)
    db.add(prank)
    db.commit()
    db.refresh(prank)

    share_url = f"/p/{prank.slug}"
    return PrankCreateResponse(prank_id=str(prank.id), slug=prank.slug, share_url=share_url)


def _validate_image_path(slug: str, relative_path: str) -> str:
    resolved = resolve_image_path(slug, relative_path)
    root = PRANK_IMAGE_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid image path") from exc

    if not resolved.exists():
        raise HTTPException(status_code=400, detail=f"Image not found at {relative_path}")
    return str(resolved)


@app.get("/api/pranks/{slug}", response_model=PrankDetailResponse)
def get_prank(slug: str, db: Session = Depends(get_db)) -> PrankDetailResponse:
    prank = db.query(Prank).filter(Prank.slug == slug).first()
    if prank is None:
        raise HTTPException(status_code=404, detail="Prank not found")

    triggers = (
        db.query(PrankTrigger)
        .filter(PrankTrigger.prank_id == prank.id)
        .order_by(PrankTrigger.created_at.asc())
        .all()
    )
    trigger_payloads = [
        PrankTriggerInfo(
            id=str(t.id),
            trigger_text=t.trigger_text,
            image_base64=load_prank_image_base64(t.image_path),
        )
        for t in triggers
    ]
    return PrankDetailResponse(
        prank_id=str(prank.id),
        slug=prank.slug,
        title=prank.title,
        triggers=trigger_payloads,
    )


@app.post("/api/pranks/{prank_id}/triggers", response_model=PrankTriggerCreateResponse)
def add_prank_trigger(
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

    payload = file.file.read()
    ext = ".png"
    if file.filename and "." in file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower()

    image_path = save_prank_image(prank.slug, payload, extension=ext)

    trigger = PrankTrigger(
        prank_id=prank.id,
        trigger_text=trigger_text,
        image_path=image_path,
    )
    db.add(trigger)
    db.commit()
    db.refresh(trigger)

    return PrankTriggerCreateResponse(
        id=str(trigger.id),
        trigger_text=trigger.trigger_text,
        image_path=trigger.image_path,
    )


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
    text_request = _build_text_request(
        prompt=request.prompt,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        width=request.width,
        height=request.height,
        seed=request.seed,
    )
    if not triggers:
        return _run_flux(text_request)

    trap_texts = [t.trigger_text for t in triggers]
    idx = choose_matching_trigger(request.prompt, trap_texts)

    if idx is not None:
        trigger = triggers[idx]
        image_base64 = load_prank_image_base64(trigger.image_path)
        return ImageResponse(image_base64=image_base64)

    return _run_flux(text_request)
