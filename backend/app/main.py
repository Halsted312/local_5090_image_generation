from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
import secrets
import string
import threading
import time
from typing import Iterable

import torch
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi import File, Form, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from pathlib import Path
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .flux_models import (
    get_logo_pipeline,
    get_realvis_pipeline,
    get_sd3_pipeline,
    get_text_pipeline,
)
from .models import GenerationLog, Prank, PrankTrigger
from .prank_matching import get_prank_matcher_llm, match_prank_trigger
from .model_registry import MODEL_REGISTRY
from .router_engine import RoutingDecision, route_prompt
from .schemas import (
    ImageResponse,
    PrankCreateResponse,
    PrankDetailResponse,
    PrankGenerateRequest,
    MatchedTrigger,
    PrankMetadataCreate,
    PrankTriggerCreateResponse,
    PrankTriggerInfo,
    PrankTriggerUpdateRequest,
    PrankSummary,
    RoutingMetadata,
    TextGenerateRequest,
    GenerationLogEntry,
    AdminLoginRequest,
    AdminLoginResponse,
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

PRANK_DELAY_MS = int(os.getenv("PRANK_DELAY_MS", "0") or "0")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
ADMIN_TOKEN = secrets.token_urlsafe(32)
ADMIN_TOKEN_ISSUED = time.time()
ADMIN_TOKEN_TTL_SECONDS = int(os.getenv("ADMIN_TOKEN_TTL_SECONDS", str(24 * 3600)) or str(24 * 3600))
ADMIN_LOGIN_WINDOW = int(os.getenv("ADMIN_LOGIN_WINDOW", "60") or "60")
ADMIN_LOGIN_MAX_ATTEMPTS = int(os.getenv("ADMIN_LOGIN_MAX_ATTEMPTS", "10") or "10")

ALLOWED_ORIGINS: Iterable[str] = (
    "https://promptpics.ai",
    "https://www.promptpics.ai",
    "https://app.promptpics.ai",
    "https://promptpics.replit.app",
    "http://localhost:5173",
    "http://localhost:3000",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    # Allow dynamic Replit preview domains
    allow_origin_regex=r"https://.*\.replit\.app",
    allow_credentials=False,  # frontend sends credentials: 'omit'
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
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
        # Optional prank-matcher warmup to avoid first-request latency.
        try:
            matcher = get_prank_matcher_llm()
            if matcher is not None:
                matcher.choose("warmup", ["warmup"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Prank matcher LLM warmup failed: %s", exc)
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


def _sleep_prank_delay() -> None:
    """Artificial delay for prank matches to mimic generation time."""
    if PRANK_DELAY_MS > 0:
        time.sleep(PRANK_DELAY_MS / 1000.0)


def require_admin(x_admin_token: str = Header(..., alias="X-Admin-Token")) -> None:
    # Expire tokens after TTL; force re-login.
    if (time.time() - ADMIN_TOKEN_ISSUED) > ADMIN_TOKEN_TTL_SECONDS:
        raise HTTPException(status_code=401, detail="Admin token expired; please login again")
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Admins only")


@app.middleware("http")
async def block_dotfiles(request: Request, call_next):
    """
    Deny access to dotfiles and VCS paths (e.g., /.git) defensively.
    """
    path = request.url.path
    if path.startswith("/.") or "/.git" in path:
        logger.warning("Blocked dotfile/VCS path: path=%s ip=%s", path, _client_ip(request))
        return Response(status_code=404)
    # Light logging for obviously suspicious probes
    for marker in ("/wp-login", "/wp-admin", "/phpmyadmin", "/config.php", "/etc/passwd"):
        if marker in path:
            logger.warning("Suspicious probe blocked: path=%s ip=%s", path, _client_ip(request))
            return Response(status_code=404)
    return await call_next(request)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _check_admin_rate_limit(request: Request) -> None:
    """
    Simple in-memory rate limiter per client IP for admin login.
    """
    ip = _client_ip(request)
    now = time.time()
    attempts = getattr(app.state, "admin_login_attempts", None)
    if attempts is None:
        attempts = {}
        app.state.admin_login_attempts = attempts
    window = ADMIN_LOGIN_WINDOW
    limit = ADMIN_LOGIN_MAX_ATTEMPTS
    history = [t for t in attempts.get(ip, []) if now - t < window]
    if len(history) >= limit:
        raise HTTPException(status_code=429, detail="Too many admin login attempts; try again later")
    history.append(now)
    attempts[ip] = history


def _issue_admin_token() -> str:
    global ADMIN_TOKEN, ADMIN_TOKEN_ISSUED
    ADMIN_TOKEN = secrets.token_urlsafe(32)
    ADMIN_TOKEN_ISSUED = time.time()
    return ADMIN_TOKEN


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
    start = time.time()
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
    generation_time_ms = int((time.time() - start) * 1000)

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
        model_used=actual_model,
        thumbnail_base64=_encode_thumbnail_base64(thumb_path),
        image_path=image_path,
        thumbnail_path=thumb_path,
        router_metadata=_routing_metadata(decision),
        was_prank=False,
        matched_trigger_id=None,
        matched_trigger_text=None,
        generation_time_ms=generation_time_ms,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/admin/login", response_model=AdminLoginResponse)
def admin_login(payload: AdminLoginRequest, request: Request) -> AdminLoginResponse:
    # rate-limit attempts per client
    _check_admin_rate_limit(request)
    if not ADMIN_PASSWORD:
        raise HTTPException(status_code=500, detail="Admin password not configured")
    if payload.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid admin password")
    token = _issue_admin_token()
    return AdminLoginResponse(admin_token=token)


@app.post("/api/admin/verify")
def admin_verify(_: None = Depends(require_admin)) -> dict:
    """Simple admin token verification endpoint."""
    return {"status": "ok"}


@app.options("/api/generate")
async def generate_options() -> Response:
    """Explicit preflight handler; CORSMiddleware adds headers."""
    return Response(status_code=200)


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


def _prank_to_summary(prank: Prank, trigger_count: int) -> dict:
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
        "triggerCount": trigger_count,
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


@app.get("/api/pranks/slug/{slug}", response_model=PrankDetailResponse)
def get_prank_alias(slug: str, db: Session = Depends(get_db)) -> PrankDetailResponse:
    """Alias for GET /api/pranks/{slug} to match alternate caller expectations."""
    return get_prank(slug, db)


@app.get("/api/pranks", response_model=list[PrankDetailResponse])
def list_pranks(
    session_id: str | None = Query(None, alias="sessionId"),
    db: Session = Depends(get_db),
) -> list[PrankDetailResponse]:
    """
    List prank sets owned by the provided session_id. Returns [] when session_id is missing.
    """
    if not session_id:
        return []

    pranks = (
        db.query(Prank)
        .filter(Prank.session_id == session_id)
        .order_by(Prank.created_at.desc())
        .all()
    )

    results: list[PrankDetailResponse] = []
    for prank in pranks:
        triggers = (
            db.query(PrankTrigger)
            .filter(PrankTrigger.prank_id == prank.id)
            .order_by(PrankTrigger.created_at.asc())
            .all()
        )
        payload = _prank_to_response(prank, triggers)
        results.append(PrankDetailResponse(**payload))
    return results


@app.get("/api/pranks/summaries", response_model=list[PrankSummary])
def list_prank_summaries(
    session_id: str | None = Query(None, alias="sessionId"),
    db: Session = Depends(get_db),
) -> list[PrankSummary]:
    """
    Lightweight prank summaries without base64 payloads.
    """
    if not session_id:
        return []

    pranks = (
        db.query(Prank)
        .filter(Prank.session_id == session_id)
        .order_by(Prank.created_at.desc())
        .all()
    )

    results: list[PrankSummary] = []
    for prank in pranks:
        trigger_count = (
            db.query(PrankTrigger)
            .filter(PrankTrigger.prank_id == prank.id)
            .count()
        )
        payload = _prank_to_summary(prank, trigger_count)
        results.append(PrankSummary(**payload))
    return results


@app.get("/api/admin/pranks", response_model=list[PrankSummary])
def admin_list_pranks(
    _: None = Depends(require_admin),
    db: Session = Depends(get_db),
) -> list[PrankSummary]:
    """
    Admin-only list of all pranks (summary only, no base64 payloads).
    """
    pranks = (
        db.query(Prank)
        .order_by(Prank.created_at.desc())
        .all()
    )

    results: list[PrankSummary] = []
    for prank in pranks:
        trigger_count = (
            db.query(PrankTrigger)
            .filter(PrankTrigger.prank_id == prank.id)
            .count()
        )
        payload = _prank_to_summary(prank, trigger_count)
        results.append(PrankSummary(**payload))
    return results


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

    # Use share_slug for storage folder consistency
    image_path, thumb_path = save_prank_image_with_thumbnail(
        prank.share_slug or prank.slug or "prank", payload, extension=ext
    )

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


@app.post("/api/pranks/{slug}/triggers", response_model=PrankTriggerCreateResponse)
def add_prank_trigger_by_slug(
    slug: str,
    session_id: str | None = Form(None, alias="sessionId"),
    trigger_text: str = Form(..., alias="triggerText"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> PrankTriggerCreateResponse:
    prank = db.query(Prank).filter(
        (Prank.share_slug == slug) | (Prank.builder_slug == slug) | (Prank.slug == slug)
    ).first()
    if prank is None:
        raise HTTPException(status_code=404, detail="Prank not found.")
    # Enforce ownership when session_id is provided
    if prank.session_id and session_id and prank.session_id != session_id:
        raise HTTPException(status_code=403, detail="Not authorized to modify this prank.")
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
        engine=request.engine or "auto",
        session_id=request.session_id,
    )
    if not triggers:
        return _process_generation(text_request, db, share_slug=prank.share_slug)

    trap_texts = [t.trigger_text for t in triggers]
    # Heuristic + optional LLM matcher
    matcher_llm = get_prank_matcher_llm()
    idx, debug = match_prank_trigger(request.prompt, trap_texts, llm=matcher_llm)
    start_time = time.time()
    logger.info(
        "Prank match debug for slug=%s prompt=%r: %s",
        slug,
        request.prompt,
        debug,
    )

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
            reason=f"Matched prank trigger: {trigger.trigger_text}",
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

        # Artificial delay to mimic generation time for prank hits.
        _sleep_prank_delay()
        total_ms = int((time.time() - start_time) * 1000)

        return ImageResponse(
            image_base64=load_prank_image_base64(trigger.image_path),
            generation_id=generation_id,
            model_id="prank",
            model_used="prank",
            thumbnail_base64=_encode_thumbnail_base64(thumb_path),
            image_path=image_path,
            thumbnail_path=thumb_path,
            router_metadata=_routing_metadata(decision),
            was_prank=True,
            matched_trigger_id=str(trigger.id),
            matched_trigger_text=trigger.trigger_text,
            generation_time_ms=total_ms,
            is_prank_match=True,
            matched_trigger=MatchedTrigger(
                id=str(trigger.id),
                trigger_text=trigger.trigger_text,
                image_base64=load_prank_image_base64(trigger.image_path),
                thumbnail_base64=_encode_thumbnail_base64(trigger.thumbnail_path)
                if trigger.thumbnail_path
                else None,
            ),
        )

    resp = _process_generation(text_request, db, share_slug=prank.share_slug)
    # Explicitly mark as non-prank when no trigger matched.
    resp.is_prank_match = False
    resp.matched_trigger = None
    resp.matched_trigger_id = None
    resp.matched_trigger_text = None
    resp.was_prank = False
    if resp.model_id and not resp.model_used:
        resp.model_used = resp.model_id
    return resp


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
    session_id: str | None = Query(None, alias="sessionId"),
    db: Session = Depends(get_db),
) -> PrankTriggerCreateResponse:
    prank = _get_prank_by_slug(db, slug)
    if prank.session_id and session_id and prank.session_id != session_id:
        raise HTTPException(status_code=403, detail="Not authorized to modify this prank.")
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
    session_id: str | None = Query(None, alias="sessionId"),
    db: Session = Depends(get_db),
) -> dict:
    prank = _get_prank_by_slug(db, slug)
    if prank.session_id and session_id and prank.session_id != session_id:
        raise HTTPException(status_code=403, detail="Not authorized to modify this prank.")
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


@app.post("/api/pranks/{slug}/triggers/{trigger_id}/edit", response_model=PrankTriggerCreateResponse)
def edit_prank_trigger_image(
    slug: str,
    trigger_id: str,
    session_id: str | None = Form(None, alias="sessionId"),
    image: UploadFile = File(...),
    edit_description: str | None = Form(None, alias="edit_description"),
    db: Session = Depends(get_db),
) -> PrankTriggerCreateResponse:
    prank = _get_prank_by_slug(db, slug)

    # Enforce ownership when session_id is provided.
    if prank.session_id and session_id and prank.session_id != session_id:
        raise HTTPException(status_code=403, detail="Not authorized to modify this prank.")

    trigger = (
        db.query(PrankTrigger)
        .filter(PrankTrigger.id == trigger_id, PrankTrigger.prank_id == prank.id)
        .first()
    )
    if not trigger:
        raise HTTPException(status_code=404, detail="Trigger not found")

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    payload = image.file.read()
    ext = ".png"
    if image.filename and "." in image.filename:
        ext = "." + image.filename.rsplit(".", 1)[-1].lower()

    image_path, thumb_path = save_prank_image_with_thumbnail(
        prank.share_slug or prank.slug or "prank",
        payload,
        extension=ext,
    )

    trigger.image_path = image_path
    trigger.thumbnail_path = thumb_path
    db.commit()
    db.refresh(trigger)

    return PrankTriggerCreateResponse(
        id=str(trigger.id),
        trigger_text=trigger.trigger_text,
        image_path=trigger.image_path,
        thumbnail_path=trigger.thumbnail_path,
    )


@app.get("/api/pranks/{slug}/triggers/{trigger_id}/thumbnail")
def get_prank_trigger_thumbnail(
    slug: str,
    trigger_id: str,
    db: Session = Depends(get_db),
):
    prank = _get_prank_by_slug(db, slug)
    trigger = (
        db.query(PrankTrigger)
        .filter(PrankTrigger.id == trigger_id, PrankTrigger.prank_id == prank.id)
        .first()
    )
    if not trigger:
        raise HTTPException(status_code=404, detail="Trigger not found")

    thumb_path = trigger.thumbnail_path or trigger.image_path
    if not thumb_path:
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    path = Path(thumb_path)
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail file missing")

    return FileResponse(path, media_type=_file_media_type(path))


@app.get("/api/generations", response_model=list[GenerationLogEntry])
def list_generations(
    limit: int = 50,
    offset: int = 0,
    session_id: str | None = Query(None, alias="sessionId"),
    db: Session = Depends(get_db),
) -> list[GenerationLogEntry]:
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    # Privacy: require session_id to filter; otherwise return empty list.
    if not session_id:
        return []

    rows = (
        db.query(GenerationLog)
        .filter(GenerationLog.session_id == session_id)
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


def _file_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


@app.get("/api/images/thumb/{generation_id}")
def get_generation_thumbnail(
    generation_id: str,
    db: Session = Depends(get_db),
):
    gen = db.query(GenerationLog).filter(GenerationLog.id == generation_id).first()
    if not gen or not gen.thumbnail_path:
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    thumb_path = Path(gen.thumbnail_path)
    if not thumb_path.is_absolute():
        thumb_path = (Path(gen.thumbnail_path)).resolve()
    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail file missing")

    return FileResponse(thumb_path, media_type=_file_media_type(thumb_path))


@app.get("/api/models")
def list_models_api() -> dict:
    """
    Expose model registry for frontend to consume.
    """
    models = []
    for model_id, info in MODEL_REGISTRY.items():
        models.append(
            {
                "id": model_id,
                "display_name": info.get("display_name", model_id),
                "tags": info.get("tags", []),
                "notes": info.get("notes", ""),
            }
        )
    return {"models": models}
