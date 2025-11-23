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
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
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

# Enable TF32 for faster matmul on supported GPUs (e.g., 5090)
torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]

PRANK_DELAY_MS = int(os.getenv("PRANK_DELAY_MS", "0") or "0")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
ADMIN_TOKEN = secrets.token_urlsafe(32)
ADMIN_TOKEN_ISSUED = time.time()
ADMIN_TOKEN_TTL_SECONDS = int(os.getenv("ADMIN_TOKEN_TTL_SECONDS", str(24 * 3600)) or str(24 * 3600))
ADMIN_LOGIN_WINDOW = int(os.getenv("ADMIN_LOGIN_WINDOW", "60") or "60")
ADMIN_LOGIN_MAX_ATTEMPTS = int(os.getenv("ADMIN_LOGIN_MAX_ATTEMPTS", "10") or "10")
REWRITE_MODEL_ID = os.getenv("REWRITE_LLM_ID", os.getenv("ROUTER_LLM_ID", "Qwen/Qwen2.5-1.5B-Instruct"))
_rewrite_pipeline = None
_rewrite_lock = threading.Lock()
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "768") or "768")
GPU_COORD_PATH_ENV = os.getenv("GPU_COORD_PATH", "").strip()
GPU_COORD_PATH = Path(GPU_COORD_PATH_ENV) if GPU_COORD_PATH_ENV else None
GPU_COORD_GRACE_SECONDS = int(os.getenv("GPU_COORD_GRACE_SECONDS", "30") or "30")

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
    allow_origins=list(ALLOWED_ORIGINS),  # Only explicit origins, no wildcards
    allow_credentials=False,  # frontend sends credentials: 'omit'
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

SLUG_ALPHABET = string.ascii_letters + string.digits


@app.on_event("startup")
def _ensure_tables() -> None:
    """Create tables on startup for local/dev usage."""
    try:
        # Clear CUDA cache on startup for clean memory state
        if torch.cuda.is_available():
            logger.info("Clearing CUDA cache on startup...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            logger.info(f"GPU Memory: {free_mem:.2f} GB free / {total_mem:.2f} GB total")

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
    # Handle "meta" device case when using device_map="balanced"
    device_str = str(device)
    if device_str == "meta":
        device_str = "cpu"
    return torch.Generator(device=device_str).manual_seed(seed)


def _free_cuda_memory() -> None:
    """Best-effort VRAM cleanup after requests to reduce OOM risk."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001
            pass
        try:
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass


def _read_gpu_coord_owner() -> tuple[str | None, float | None]:
    """Read current GPU owner and timestamp from coordination file."""
    if not GPU_COORD_PATH or not GPU_COORD_PATH.exists():
        return None, None
    try:
        data = json.loads(GPU_COORD_PATH.read_text())
        return data.get("owner"), float(data.get("since", 0.0))
    except Exception:  # noqa: BLE001
        return None, None


def _acquire_gpu_coord(owner: str) -> None:
    """Mark GPU as owned by `owner`."""
    if not GPU_COORD_PATH:
        return
    try:
        GPU_COORD_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"owner": owner, "since": time.time()})
        tmp_path = GPU_COORD_PATH.with_suffix(".tmp")
        tmp_path.write_text(payload)
        tmp_path.replace(GPU_COORD_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to acquire GPU coord for %s: %s", owner, exc)


def _release_gpu_coord(owner: str) -> None:
    """Clear ownership if we still hold it."""
    if not GPU_COORD_PATH or not GPU_COORD_PATH.exists():
        return
    try:
        data = json.loads(GPU_COORD_PATH.read_text())
        if data.get("owner") != owner:
            return
    except Exception:
        pass
    try:
        GPU_COORD_PATH.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to release GPU coord for %s: %s", owner, exc)


def _wait_for_gpu(owner: str) -> None:
    """If another owner holds the GPU, wait briefly for release before proceeding."""
    if not GPU_COORD_PATH or not GPU_COORD_PATH.exists():
        return
    start = time.time()
    while GPU_COORD_PATH.exists():
        current_owner, since = _read_gpu_coord_owner()
        if current_owner in (None, owner):
            break
        elapsed = time.time() - start
        if elapsed > GPU_COORD_GRACE_SECONDS:
            logger.warning(
                "GPU coord owned by %s for %.1fs; proceeding for owner %s",
                current_owner,
                elapsed,
                owner,
            )
            break
        time.sleep(0.25)


def _generate_unique_slug(db: Session, length: int = 5) -> str:
    """Generate a unique slug, avoiding reserved slugs like 'imagine'."""
    RESERVED_SLUGS = {"imagine"}  # Reserved for VIP pranks

    while True:
        candidate = "".join(random.choices(SLUG_ALPHABET, k=length))

        # Skip reserved slugs
        if candidate.lower() in RESERVED_SLUGS:
            continue

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


def _get_rewrite_pipeline():
    """Load or return the small LLM used to clean NSFW prompts."""
    global _rewrite_pipeline
    if _rewrite_pipeline is None:
        with _rewrite_lock:
            if _rewrite_pipeline is None:
                logger.info("Loading rewrite LLM: %s", REWRITE_MODEL_ID)
                _rewrite_pipeline = pipeline(
                    "text-generation",
                    model=AutoModelForCausalLM.from_pretrained(
                        REWRITE_MODEL_ID,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                    ),
                    tokenizer=AutoTokenizer.from_pretrained(REWRITE_MODEL_ID),
                )
    return _rewrite_pipeline


def _rewrite_prompt_safe(prompt: str) -> str | None:
    """
    Ask small LLM to make prompt non-NSFW while preserving meaning.
    Returns rewritten prompt or None on failure.
    """
    system = (
        "The prompt below was flagged as NSFW by an image model, but the intent is safe. "
        "Rewrite it to be clearly safe-for-work while preserving meaning. "
        "Disambiguate any words that might imply drugs or explicit content "
        "(e.g., 'weed' -> 'grass', 'burning grass' -> 'controlled burn of vegetation'). "
        "Keep it concise; do not add new concepts."
    )
    try:
        pipe = _get_rewrite_pipeline()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Rewrite LLM unavailable: %s", exc)
        return None
    try:
        outputs = pipe(
            f"{system}\nPrompt: {prompt}\nRewritten:",
            max_new_tokens=128,
            do_sample=False,
            temperature=0.3,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        text = outputs[0]["generated_text"]
        # Heuristic: take the substring after 'Rewritten:' if present
        marker = "Rewritten:"
        if marker in text:
            return text.split(marker, 1)[1].strip()
        return text.strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Rewrite LLM call failed: %s", exc)
        return None


def _is_black_image(image: Image.Image, threshold: int = 2) -> bool:
    """
    Detect if an image is essentially black (safety checker output).
    threshold: max pixel value allowed to still count as black.
    """
    extrema = image.getextrema()
    if isinstance(extrema[0], tuple):
        # Multichannel
        return all(channel_max <= threshold for _, channel_max in extrema)
    # Single channel
    return extrema[1] <= threshold


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


@app.get("/gpu-status")
def gpu_status() -> dict:
    """Lightweight coordination endpoint for benchmark runner."""
    owner, since = _read_gpu_coord_owner()
    busy = owner is not None
    return {
        "busy": busy,
        "owner": owner,
        "since": since,
        "grace_seconds": GPU_COORD_GRACE_SECONDS,
        "path": str(GPU_COORD_PATH) if GPU_COORD_PATH else None,
    }


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
    # Clear any lingering CUDA allocations before selecting a pipeline.
    _free_cuda_memory()
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

    # Use HiDream-specific defaults when logo_sdxl is selected
    if model_id == "logo_sdxl":
        from .config import HIDREAM_STEPS, HIDREAM_GUIDANCE
        num_steps = request.num_inference_steps or HIDREAM_STEPS
        guidance = request.guidance_scale if request.guidance_scale is not None else HIDREAM_GUIDANCE
    else:
        num_steps = request.num_inference_steps
        guidance = request.guidance_scale

    # Get device, handling special cases for CPU offload models
    device = getattr(pipe, "device", "cpu")
    # When using device_map or CPU offload, device might be "meta" - use "cpu" for generator
    if str(device) == "meta":
        generator_device = "cpu"
    else:
        generator_device = device
    generator = _make_generator(generator_device, request.seed)
    with generation_lock:
        _wait_for_gpu(owner="backend")
        _acquire_gpu_coord(owner="backend")
        try:
            result = pipe(
                prompt=request.prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                width=request.width,
                height=request.height,
                generator=generator,
            )
            image = result.images[0]
            nsfw = getattr(result, "nsfw_content_detected", None)
            if nsfw is not None:
                logger.info("NSFW check for model %s prompt %r: %s", selected, request.prompt, nsfw)
            # If flagged NSFW, attempt to rewrite prompt and retry once.
            nsfw_flag = False
            if isinstance(nsfw, list):
                nsfw_flag = any(nsfw)
            elif isinstance(nsfw, bool):
                nsfw_flag = nsfw
            if nsfw_flag or _is_black_image(image):
                rewritten = _rewrite_prompt_safe(request.prompt)
                if rewritten and rewritten != request.prompt:
                    logger.info("Retrying generation for %s with rewritten safe prompt: %r -> %r", selected, request.prompt, rewritten)
                    result = pipe(
                        prompt=rewritten,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance,
                        width=request.width,
                        height=request.height,
                        generator=_make_generator(generator_device, request.seed),
                    )
                    image = result.images[0]
                    nsfw_retry = getattr(result, "nsfw_content_detected", None)
                    if nsfw_retry is not None:
                        logger.info("NSFW check after rewrite for %s: %s", selected, nsfw_retry)
                    if _is_black_image(image):
                        logger.warning("Image still black after rewrite for %s", selected)
                        if model_id == "logo_sdxl":
                            # Last-resort fallback to flux_dev to avoid returning black image.
                            pipe = get_text_pipeline()
                            selected = "flux_dev"
                            fallback_device = getattr(pipe, "device", "cpu")
                            fallback_generator = _make_generator(fallback_device if str(fallback_device) != "meta" else "cpu", request.seed)
                            result = pipe(
                                prompt=rewritten,
                                num_inference_steps=request.num_inference_steps,
                                guidance_scale=request.guidance_scale,
                                width=request.width,
                                height=request.height,
                                generator=fallback_generator,
                            )
                            image = result.images[0]
        except Exception as exc:  # noqa: BLE001
            # If HiDream/logo path fails, fall back to flux_dev to return something instead of 500.
            if model_id == "logo_sdxl":
                logger.exception("Logo/HiDream generation failed, falling back to flux_dev")
                try:
                    pipe = get_text_pipeline()
                    selected = "flux_dev"
                    fallback_device = getattr(pipe, "device", "cpu")
                    fallback_generator = _make_generator(fallback_device if str(fallback_device) != "meta" else "cpu", request.seed)
                    result = pipe(
                        prompt=request.prompt,
                        num_inference_steps=request.num_inference_steps,
                        guidance_scale=request.guidance_scale,
                        width=request.width,
                        height=request.height,
                        generator=fallback_generator,
                    )
                    image = result.images[0]
                except Exception as fallback_exc:  # noqa: BLE001
                    logger.exception("Fallback to flux_dev also failed")
                    raise HTTPException(status_code=500, detail="Image generation failed") from fallback_exc
            else:
                logger.exception("Image generation failed for model %s", selected)
                raise HTTPException(status_code=500, detail="Image generation failed") from exc
        finally:
            _release_gpu_coord(owner="backend")
            _free_cuda_memory()
            # Brief pause to let CUDA allocator settle before next owner.
            time.sleep(0.5)

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
    # Clamp resolution for performance and VRAM headroom.
    if request.width and request.width > MAX_IMAGE_SIDE:
        request.width = MAX_IMAGE_SIDE
    if request.height and request.height > MAX_IMAGE_SIDE:
        request.height = MAX_IMAGE_SIDE

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


@app.post("/api/admin/pranks/vip", response_model=PrankCreateResponse)
def get_or_create_vip_prank(
    _: None = Depends(require_admin),
    db: Session = Depends(get_db)
) -> PrankCreateResponse:
    """
    Idempotent endpoint that ensures the VIP 'imagine' prank exists and returns its details.
    Only admin can call this endpoint.
    """
    # Check if the VIP prank already exists
    existing_prank = db.query(Prank).filter(Prank.share_slug == "imagine").first()

    if existing_prank:
        # Return existing VIP prank
        triggers = list(existing_prank.triggers)
        payload = _prank_to_response(existing_prank, triggers)
        return PrankCreateResponse(**payload)

    # Create the VIP prank
    builder_slug = _generate_unique_slug(db, length=8)
    vip_prank = Prank(
        share_slug="imagine",
        builder_slug=builder_slug,
        slug="imagine",  # legacy compatibility
        title="CEO VIP Prank Page",
        session_id=None,  # No session owner, admin-only
        is_vip=True,
        is_admin_only=True,
        view_count=0
    )

    db.add(vip_prank)
    db.commit()
    db.refresh(vip_prank)

    payload = _prank_to_response(vip_prank, [])
    return PrankCreateResponse(**payload)


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

    # Special URL for VIP prank
    if share_slug == "imagine":
        share_url = f"{base_url}/imagine"
    else:
        share_url = f"{base_url}/p/{share_slug}"

    return {
        "id": str(prank.id),
        "slug": share_slug,  # legacy field
        "shareSlug": share_slug,  # explicit shareSlug field
        "builderSlug": builder_slug,
        "title": prank.title,
        "sessionId": prank.session_id,
        "shareUrl": share_url,
        "builderUrl": f"{base_url}/customize/{builder_slug}",
        "createdAt": prank.created_at.isoformat() if prank.created_at else None,
        "viewCount": prank.view_count or 0,
        "isVip": prank.is_vip if hasattr(prank, 'is_vip') else False,
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

    # Special URL for VIP prank
    if share_slug == "imagine":
        share_url = f"{base_url}/imagine"
    else:
        share_url = f"{base_url}/p/{share_slug}"

    return {
        "id": str(prank.id),
        "slug": share_slug,  # legacy field
        "shareSlug": share_slug,  # explicit shareSlug field
        "builderSlug": builder_slug,
        "title": prank.title,
        "sessionId": prank.session_id,
        "shareUrl": share_url,
        "builderUrl": f"{base_url}/customize/{builder_slug}",
        "createdAt": prank.created_at.isoformat() if prank.created_at else None,
        "viewCount": prank.view_count or 0,
        "triggerCount": trigger_count,
        "isVip": prank.is_vip if hasattr(prank, 'is_vip') else False,
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
    x_admin_token: str | None = Header(None, alias="X-Admin-Token"),
    db: Session = Depends(get_db),
) -> PrankTriggerCreateResponse:
    prank = db.query(Prank).filter(
        (Prank.share_slug == slug) | (Prank.builder_slug == slug) | (Prank.slug == slug)
    ).first()
    if prank is None:
        raise HTTPException(status_code=404, detail="Prank not found.")

    # Check if prank is admin-only
    if hasattr(prank, 'is_admin_only') and prank.is_admin_only:
        # Admin-only pranks require valid admin token
        if not x_admin_token or x_admin_token != ADMIN_TOKEN:
            raise HTTPException(status_code=403, detail="Admin access required for this prank.")
        if (time.time() - ADMIN_TOKEN_ISSUED) > ADMIN_TOKEN_TTL_SECONDS:
            raise HTTPException(status_code=401, detail="Admin token expired; please login again")
    else:
        # Regular prank: enforce ownership when session_id is provided
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
    x_admin_token: str | None = Header(None, alias="X-Admin-Token"),
    db: Session = Depends(get_db),
) -> PrankTriggerCreateResponse:
    prank = _get_prank_by_slug(db, slug)

    # Check if prank is admin-only
    if hasattr(prank, 'is_admin_only') and prank.is_admin_only:
        # Admin-only pranks require valid admin token
        if not x_admin_token or x_admin_token != ADMIN_TOKEN:
            raise HTTPException(status_code=403, detail="Admin access required for this prank.")
        if (time.time() - ADMIN_TOKEN_ISSUED) > ADMIN_TOKEN_TTL_SECONDS:
            raise HTTPException(status_code=401, detail="Admin token expired; please login again")
    else:
        # Regular prank: enforce ownership when session_id is provided
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
    x_admin_token: str | None = Header(None, alias="X-Admin-Token"),
    db: Session = Depends(get_db),
) -> dict:
    prank = _get_prank_by_slug(db, slug)

    # Check if prank is admin-only
    if hasattr(prank, 'is_admin_only') and prank.is_admin_only:
        # Admin-only pranks require valid admin token
        if not x_admin_token or x_admin_token != ADMIN_TOKEN:
            raise HTTPException(status_code=403, detail="Admin access required for this prank.")
        if (time.time() - ADMIN_TOKEN_ISSUED) > ADMIN_TOKEN_TTL_SECONDS:
            raise HTTPException(status_code=401, detail="Admin token expired; please login again")
    else:
        # Regular prank: enforce ownership when session_id is provided
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
    x_admin_token: str | None = Header(None, alias="X-Admin-Token"),
    db: Session = Depends(get_db),
) -> PrankTriggerCreateResponse:
    prank = _get_prank_by_slug(db, slug)

    # Check if prank is admin-only
    if hasattr(prank, 'is_admin_only') and prank.is_admin_only:
        # Admin-only pranks require valid admin token
        if not x_admin_token or x_admin_token != ADMIN_TOKEN:
            raise HTTPException(status_code=403, detail="Admin access required for this prank.")
        if (time.time() - ADMIN_TOKEN_ISSUED) > ADMIN_TOKEN_TTL_SECONDS:
            raise HTTPException(status_code=401, detail="Admin token expired; please login again")
    else:
        # Regular prank: enforce ownership when session_id is provided
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
