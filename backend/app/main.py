from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
import string
import threading
from typing import Iterable

import torch
from fastapi import Depends, FastAPI, HTTPException
from fastapi import File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .flux_models import (
    get_logo_pipeline,
    get_realvis_pipeline,
    get_sd3_pipeline,
    get_text_pipeline,
)
from .llm_matcher import choose_matching_trigger
from .models import GenerationLog, Prank, PrankTrigger
from .router_engine import RoutingDecision, route_prompt
from .schemas import (
    ImageResponse,
    PrankCreateResponse,
    PrankDetailResponse,
    PrankGenerateRequest,
    PrankMetadataCreate,
    PrankTriggerCreateResponse,
    PrankTriggerInfo,
    PrankTriggerUpdateRequest,
    RoutingMetadata,
    TextGenerateRequest,
    GenerationLogEntry,
)
from .storage import (
    GEN_IMAGE_ROOT,
    GEN_THUMB_ROOT,
    PRANK_IMAGE_ROOT,
    load_prank_image_base64,
    resolve_image_path,
    save_prank_image_with_thumbnail,
    save_generation_image,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FLUX Image API + Prank Mode")

ALLOWED_ORIGINS: Iterable[str] = (
    "https://promptpics.ai",
    "https://*.replit.app",
    "http://localhost:3000",
    "https://app.promptpics.ai",
)

app.add_middleware(
    CORSMiddleware,
    # Using wildcard to support dynamic Replit origins; tighten later if needed.
    allow_origins=["*"],
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
        # Global lock to avoid concurrent GPU generations.
        app.state.generation_lock = threading.Lock()
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
        existing = (
            db.query(Prank)
            .filter(
                (Prank.share_slug == candidate)
                | (Prank.builder_slug == candidate)
                | (Prank.slug == candidate)
            )
            .first()
        )
        if existing is None:
            return candidate


def _get_prank_by_slug(db: Session, slug: str) -> Prank:
    prank = (
        db.query(Prank)
        .filter((Prank.share_slug == slug) | (Prank.builder_slug == slug) | (Prank.slug == slug))
        .first()
    )
    if prank is None:
        raise HTTPException(status_code=404, detail="Prank not found")
    return prank


def _build_text_request(
    prompt: str,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    engine: str = "auto",
    session_id: str | None = None,
) -> TextGenerateRequest:
    return TextGenerateRequest(
        prompt=prompt,
        num_inference_steps=4 if num_inference_steps is None else num_inference_steps,
        guidance_scale=0.0 if guidance_scale is None else guidance_scale,
        width=1024 if width is None else width,
        height=1024 if height is None else height,
        seed=seed,
        engine=engine,
        session_id=session_id,
    )


def _routing_metadata(decision: RoutingDecision) -> RoutingMetadata:
    return RoutingMetadata(
        chosen_model_id=decision.chosen_model_id,
        scores=decision.scores,
        tags=decision.tags,
        reason=decision.reason,
    )


def _execute_model(model_id: str, request: TextGenerateRequest) -> tuple[Image.Image, str]:
    """Execute the selected model, branching to the appropriate pipeline."""
    selected = model_id
    generation_lock = app.state.generation_lock  # type: ignore[attr-defined]
    if model_id == "flux_dev":
        pipe = get_text_pipeline()
    elif model_id == "realvis_xl":
        pipe = get_realvis_pipeline()
    elif model_id == "sd3_medium":
        pipe = get_sd3_pipeline()
    elif model_id == "logo_sdxl":
        pipe = get_logo_pipeline()
    else:
        logger.warning("Model %s not implemented; falling back to flux_dev", model_id)
        pipe = get_text_pipeline()
        selected = "flux_dev"

    device = getattr(pipe, "device", "cpu")
    generator = _make_generator(device, request.seed)
    with generation_lock:
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
            logger.exception("Image generation failed for model %s", selected)
            raise HTTPException(status_code=500, detail="Image generation failed") from exc
        finally:
            _free_cuda_memory()

    return image, selected


def _log_generation(
    db: Session,
    prompt: str,
    model_id: str,
    router_decision: RoutingDecision | None,
    image_path: str,
    thumbnail_path: str,
    prank_id: str | None = None,
    share_slug: str | None = None,
    session_id: str | None = None,
) -> str:
    log_entry = GenerationLog(
        prompt=prompt,
        model_id=model_id,
        router_json=json.dumps(router_decision.__dict__) if router_decision else None,
        image_path=image_path,
        thumbnail_path=thumbnail_path,
        prank_id=prank_id,
        share_slug=share_slug,
        session_id=session_id,
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)
    return str(log_entry.id)


def _encode_thumbnail_base64(thumb_path: str) -> str | None:
    try:
        with open(thumb_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:  # noqa: BLE001
        return None


def _process_generation(
    request: TextGenerateRequest,
    db: Session,
    share_slug: str | None = None,
    prank_id: str | None = None,
) -> ImageResponse:
    if request.engine == "auto":
        decision = route_prompt(request.prompt)
        chosen_model = decision.chosen_model_id
    else:
        chosen_model = request.engine
        decision = RoutingDecision(
            chosen_model_id=chosen_model,
            scores={chosen_model: 1.0},
            tags=["manual"],
            reason="Manual engine override",
        )

    image, actual_model = _execute_model(chosen_model, request)
    image_path, thumb_path = save_generation_image(image)

    generation_id = _log_generation(
        db=db,
        prompt=request.prompt,
        model_id=actual_model,
        router_decision=decision,
        image_path=image_path,
        thumbnail_path=thumb_path,
        prank_id=prank_id,
        share_slug=share_slug,
        session_id=request.session_id,
    )

    return ImageResponse(
        image_base64=_pil_to_base64_png(image),
        generation_id=generation_id,
        model_id=actual_model,
        thumbnail_base64=_encode_thumbnail_base64(thumb_path),
        image_path=image_path,
        thumbnail_path=thumb_path,
        router_metadata=_routing_metadata(decision),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/generate", response_model=ImageResponse)
def generate_image(
    request: TextGenerateRequest,
    db: Session = Depends(get_db),
) -> ImageResponse:
    return _process_generation(request, db)


def _prank_to_response(prank: Prank, triggers: list[PrankTrigger]) -> dict:
    base_url = os.getenv("FRONTEND_BASE_URL", "https://promptpics.ai")
    share_slug = prank.share_slug
    builder_slug = prank.builder_slug
    return {
        "id": str(prank.id),
        "slug": share_slug,
        "builderSlug": builder_slug,
        "title": prank.title,
        "sessionId": prank.session_id,
        "shareUrl": f"{base_url}/p/{share_slug}",
        "builderUrl": f"{base_url}/customize/{builder_slug}",
        "createdAt": prank.created_at.isoformat() if prank.created_at else None,
        "viewCount": prank.view_count or 0,
        "triggers": [
            {
                "id": str(t.id),
                "triggerText": t.trigger_text,
                "imageBase64": load_prank_image_base64(t.image_path),
                "thumbnailBase64": load_prank_image_base64(t.thumbnail_path) if t.thumbnail_path else None,
                "createdAt": t.created_at.isoformat() if t.created_at else None,
                "matchCount": t.match_count or 0,
            }
            for t in triggers
        ],
    }


@app.post("/api/pranks", response_model=PrankCreateResponse)
def create_prank(
    metadata: PrankMetadataCreate,
    db: Session = Depends(get_db),
) -> PrankCreateResponse:
    share_slug = _generate_unique_slug(db, length=6)
    builder_slug = _generate_unique_slug(db, length=8)
    prank = Prank(
        share_slug=share_slug,
        builder_slug=builder_slug,
        slug=share_slug,  # legacy compatibility
        title=metadata.title,
        session_id=metadata.session_id,
    )
    db.add(prank)
    db.commit()
    db.refresh(prank)

    payload = _prank_to_response(prank, [])
    return PrankCreateResponse(**payload)


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
    prank = _get_prank_by_slug(db, slug)

    triggers = (
        db.query(PrankTrigger)
        .filter(PrankTrigger.prank_id == prank.id)
        .order_by(PrankTrigger.created_at.asc())
        .all()
    )
    payload = _prank_to_response(prank, triggers)
    return PrankDetailResponse(**payload)


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

    image_path, thumb_path = save_prank_image_with_thumbnail(prank.slug, payload, extension=ext)

    trigger = PrankTrigger(
        prank_id=prank.id,
        trigger_text=trigger_text,
        image_path=image_path,
        thumbnail_path=thumb_path,
    )
    db.add(trigger)
    db.commit()
    db.refresh(trigger)

    return PrankTriggerCreateResponse(
        id=str(trigger.id),
        trigger_text=trigger.trigger_text,
        image_path=trigger.image_path,
        thumbnail_path=trigger.thumbnail_path,
    )


def _add_prank_trigger(
    prank: Prank,
    trigger_text: str,
    file: UploadFile,
    db: Session,
) -> PrankTriggerCreateResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    payload = file.file.read()
    ext = ".png"
    if file.filename and "." in file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower()

    image_path, thumb_path = save_prank_image_with_thumbnail(prank.slug, payload, extension=ext)

    trigger = PrankTrigger(
        prank_id=prank.id,
        trigger_text=trigger_text,
        image_path=image_path,
        thumbnail_path=thumb_path,
    )
    db.add(trigger)
    db.commit()
    db.refresh(trigger)

    return PrankTriggerCreateResponse(
        id=str(trigger.id),
        trigger_text=trigger.trigger_text,
        image_path=trigger.image_path,
        thumbnail_path=trigger.thumbnail_path,
    )


@app.post("/api/pranks/slug/{slug}/triggers", response_model=PrankTriggerCreateResponse)
def add_prank_trigger_by_slug(
    slug: str,
    trigger_text: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> PrankTriggerCreateResponse:
    prank = _get_prank_by_slug(db, slug)
    return _add_prank_trigger(prank, trigger_text, file, db)


@app.post("/api/p/{slug}/generate", response_model=ImageResponse)
def generate_prank_image(
    slug: str,
    request: PrankGenerateRequest,
    db: Session = Depends(get_db),
) -> ImageResponse:
    prank = _get_prank_by_slug(db, slug)

    triggers = db.query(PrankTrigger).filter(PrankTrigger.prank_id == prank.id).all()
    prank.view_count = (prank.view_count or 0) + 1
    db.commit()

    text_request = _build_text_request(
        prompt=request.prompt,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        width=request.width,
        height=request.height,
        seed=request.seed,
        session_id=request.session_id,
    )
    if not triggers:
        return _process_generation(text_request, db, share_slug=prank.share_slug)

    trap_texts = [t.trigger_text for t in triggers]
    idx = choose_matching_trigger(request.prompt, trap_texts)

    if idx is not None:
        trigger = triggers[idx]
        # Save a copy + thumbnail into generation storage for logging consistency.
        try:
            with Image.open(trigger.image_path) as prank_img:
                image_path, thumb_path = save_generation_image(prank_img)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load prank image for slug %s", slug)
            raise HTTPException(status_code=500, detail="Failed to load prank image") from exc

        trigger.match_count = (trigger.match_count or 0) + 1
        db.commit()

        decision = RoutingDecision(
            chosen_model_id="prank",
            scores={"prank": 1.0},
            tags=["prank"],
            reason=f"Matched prank trigger {trigger.id}",
        )
        generation_id = _log_generation(
            db=db,
            prompt=request.prompt,
            model_id="prank",
            router_decision=decision,
            image_path=image_path,
            thumbnail_path=thumb_path,
            prank_id=str(prank.id),
            share_slug=prank.share_slug,
            session_id=request.session_id,
        )

        # Optional delay to mimic generation time when serving a stored prank.
        try:
            import time as _time  # local import to avoid global namespace noise

            _time.sleep(3)
        except Exception:
            pass

        return ImageResponse(
            image_base64=load_prank_image_base64(trigger.image_path),
            generation_id=generation_id,
            model_id="prank",
            thumbnail_base64=_encode_thumbnail_base64(thumb_path),
            image_path=image_path,
            thumbnail_path=thumb_path,
            router_metadata=_routing_metadata(decision),
        )

    return _process_generation(text_request, db, share_slug=prank.share_slug)


@app.patch("/api/pranks/{prank_id}/triggers/{trigger_id}", response_model=PrankTriggerCreateResponse)
def update_prank_trigger(
    prank_id: str,
    trigger_id: str,
    payload: PrankTriggerUpdateRequest,
    db: Session = Depends(get_db),
) -> PrankTriggerCreateResponse:
    trigger = (
        db.query(PrankTrigger)
        .filter(PrankTrigger.id == trigger_id, PrankTrigger.prank_id == prank_id)
        .first()
    )
    if not trigger:
        raise HTTPException(status_code=404, detail="Trigger not found")

    trigger.trigger_text = payload.trigger_text
    db.commit()
    db.refresh(trigger)
    return PrankTriggerCreateResponse(
        id=str(trigger.id),
        trigger_text=trigger.trigger_text,
        image_path=trigger.image_path,
        thumbnail_path=trigger.thumbnail_path,
    )


@app.patch("/api/pranks/slug/{slug}/triggers/{trigger_id}", response_model=PrankTriggerCreateResponse)
def update_prank_trigger_by_slug(
    slug: str,
    trigger_id: str,
    payload: PrankTriggerUpdateRequest,
    db: Session = Depends(get_db),
) -> PrankTriggerCreateResponse:
    prank = _get_prank_by_slug(db, slug)
    trigger = (
        db.query(PrankTrigger)
        .filter(PrankTrigger.id == trigger_id, PrankTrigger.prank_id == prank.id)
        .first()
    )
    if not trigger:
        raise HTTPException(status_code=404, detail="Trigger not found")

    trigger.trigger_text = payload.trigger_text
    db.commit()
    db.refresh(trigger)
    return PrankTriggerCreateResponse(
        id=str(trigger.id),
        trigger_text=trigger.trigger_text,
        image_path=trigger.image_path,
        thumbnail_path=trigger.thumbnail_path,
    )


@app.delete("/api/pranks/{prank_id}/triggers/{trigger_id}")
def delete_prank_trigger(
    prank_id: str,
    trigger_id: str,
    db: Session = Depends(get_db),
) -> dict:
    trigger = (
        db.query(PrankTrigger)
        .filter(PrankTrigger.id == trigger_id, PrankTrigger.prank_id == prank_id)
        .first()
    )
    if not trigger:
        raise HTTPException(status_code=404, detail="Trigger not found")

    db.delete(trigger)
    db.commit()
    return {"detail": "Trigger deleted"}


@app.delete("/api/pranks/slug/{slug}/triggers/{trigger_id}")
def delete_prank_trigger_by_slug(
    slug: str,
    trigger_id: str,
    db: Session = Depends(get_db),
) -> dict:
    prank = _get_prank_by_slug(db, slug)
    trigger = (
        db.query(PrankTrigger)
        .filter(PrankTrigger.id == trigger_id, PrankTrigger.prank_id == prank.id)
        .first()
    )
    if not trigger:
        raise HTTPException(status_code=404, detail="Trigger not found")

    db.delete(trigger)
    db.commit()
    return {"detail": "Trigger deleted"}


@app.get("/api/generations", response_model=list[GenerationLogEntry])
def list_generations(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> list[GenerationLogEntry]:
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    rows = (
        db.query(GenerationLog)
        .order_by(GenerationLog.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [
        GenerationLogEntry(
            id=str(r.id),
            prompt=r.prompt,
            model_id=r.model_id,
            image_path=r.image_path,
            thumbnail_path=r.thumbnail_path,
            created_at=r.created_at.isoformat(),
            share_slug=r.share_slug,
            router_json=json.loads(r.router_json) if r.router_json else None,
            session_id=r.session_id,
        )
        for r in rows
    ]
