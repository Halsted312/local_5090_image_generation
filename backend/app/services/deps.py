"""Shared dependencies and utilities used across routers."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from fastapi import HTTPException, Request
from PIL import Image
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from ..models import Prank, PrankTrigger

from ..database import SessionLocal
from ..queue_manager import GenerationQueue
from ..router_engine import RoutingDecision
from ..schemas import RoutingMetadata, TextGenerateRequest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------

PRANK_DELAY_MS = int(os.getenv("PRANK_DELAY_MS", "0") or "0")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
ADMIN_TOKEN = secrets.token_urlsafe(32)
ADMIN_TOKEN_ISSUED = time.time()
ADMIN_TOKEN_TTL_SECONDS = int(os.getenv("ADMIN_TOKEN_TTL_SECONDS", str(24 * 3600)) or str(24 * 3600))
ADMIN_LOGIN_WINDOW = int(os.getenv("ADMIN_LOGIN_WINDOW", "60") or "60")
ADMIN_LOGIN_MAX_ATTEMPTS = int(os.getenv("ADMIN_LOGIN_MAX_ATTEMPTS", "10") or "10")
PERCENTILE_CACHE_TTL_SECONDS = int(os.getenv("PERCENTILE_CACHE_TTL_SECONDS", "300") or "300")
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "512") or "512")
GPU_COORD_PATH_ENV = os.getenv("GPU_COORD_PATH", "").strip()
GPU_COORD_PATH = Path(GPU_COORD_PATH_ENV) if GPU_COORD_PATH_ENV else None
GPU_COORD_GRACE_SECONDS = int(os.getenv("GPU_COORD_GRACE_SECONDS", "30") or "30")
GEN_QUEUE_CAPACITY = int(os.getenv("GEN_QUEUE_CAPACITY", "64") or "64")

SLUG_ALPHABET = string.ascii_letters + string.digits
GEN_QUEUE = GenerationQueue(capacity=GEN_QUEUE_CAPACITY)
BENCH_WARMUP_PROMPT = "show me a house on a snowy hill"

# Fast, model-specific preset steps/guidance (ignore frontend sliders for speed).
MODEL_PRESETS: dict[str, dict[str, float]] = {
    "flux_dev": {"steps": 4, "guidance": 0.0},
    "realvis_xl": {"steps": 22, "guidance": 3.5},
    "sd3_medium": {"steps": 26, "guidance": 7.0},
    "hidream_dev": {"steps": 28, "guidance": 1.0},
    "flux2_dev": {"steps": 24, "guidance": 4.0},
}

# ---------------------------------------------------------------------------
# Admin Token Management
# ---------------------------------------------------------------------------


def issue_admin_token() -> str:
    global ADMIN_TOKEN, ADMIN_TOKEN_ISSUED
    ADMIN_TOKEN = secrets.token_urlsafe(32)
    ADMIN_TOKEN_ISSUED = time.time()
    return ADMIN_TOKEN


def get_admin_token() -> str:
    return ADMIN_TOKEN


def get_admin_token_issued() -> float:
    return ADMIN_TOKEN_ISSUED


# ---------------------------------------------------------------------------
# Image Utilities
# ---------------------------------------------------------------------------


def pil_to_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def make_generator(device: torch.device | str, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    # Handle "meta" device case when using device_map="balanced"
    device_str = str(device)
    if device_str == "meta":
        device_str = "cpu"
    return torch.Generator(device=device_str).manual_seed(seed)


def is_black_image(image: Image.Image, threshold: int = 2) -> bool:
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


def encode_thumbnail_base64(thumb_path: str) -> str | None:
    try:
        with open(thumb_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def file_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


# ---------------------------------------------------------------------------
# CUDA / GPU Utilities
# ---------------------------------------------------------------------------


def free_cuda_memory() -> None:
    """Best-effort VRAM cleanup after requests to reduce OOM risk."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def set_tf32(enabled: bool) -> tuple[bool | None, bool | None]:
    prev_matmul = getattr(torch.backends.cuda.matmul, "allow_tf32", None) if torch.cuda.is_available() else None
    prev_cudnn = getattr(torch.backends.cudnn, "allow_tf32", None) if torch.cuda.is_available() else None
    try:
        torch.backends.cuda.matmul.allow_tf32 = enabled  # type: ignore[attr-defined]
        torch.backends.cudnn.allow_tf32 = enabled  # type: ignore[attr-defined]
    except Exception:
        pass
    return prev_matmul, prev_cudnn


def restore_tf32(prev_matmul, prev_cudnn) -> None:
    try:
        if prev_matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul  # type: ignore[attr-defined]
        if prev_cudnn is not None:
            torch.backends.cudnn.allow_tf32 = prev_cudnn  # type: ignore[attr-defined]
    except Exception:
        pass


def tf32_enabled() -> bool | None:
    try:
        return bool(getattr(torch.backends.cuda.matmul, "allow_tf32", None))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GPU Coordination
# ---------------------------------------------------------------------------


def read_gpu_coord_owner() -> tuple[str | None, float | None]:
    """Read current GPU owner and timestamp from coordination file."""
    if not GPU_COORD_PATH or not GPU_COORD_PATH.exists():
        return None, None
    try:
        data = json.loads(GPU_COORD_PATH.read_text())
        return data.get("owner"), float(data.get("since", 0.0))
    except Exception:
        return None, None


def acquire_gpu_coord(owner: str) -> None:
    """Mark GPU as owned by `owner`."""
    if not GPU_COORD_PATH:
        return
    try:
        GPU_COORD_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"owner": owner, "since": time.time()})
        tmp_path = GPU_COORD_PATH.with_suffix(".tmp")
        tmp_path.write_text(payload)
        tmp_path.replace(GPU_COORD_PATH)
    except Exception as exc:
        logger.warning("Failed to acquire GPU coord for %s: %s", owner, exc)


def release_gpu_coord(owner: str) -> None:
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
    except Exception as exc:
        logger.warning("Failed to release GPU coord for %s: %s", owner, exc)


def wait_for_gpu(owner: str) -> None:
    """If another owner holds the GPU, wait briefly for release before proceeding."""
    if not GPU_COORD_PATH or not GPU_COORD_PATH.exists():
        return
    start = time.time()
    while GPU_COORD_PATH.exists():
        current_owner, since = read_gpu_coord_owner()
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


# ---------------------------------------------------------------------------
# Slug and Prank Utilities
# ---------------------------------------------------------------------------


def generate_unique_slug(db: Session, length: int = 5) -> str:
    """Generate a unique slug, avoiding reserved slugs like 'imagine'."""
    from ..models import Prank
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


def get_prank_by_slug(db: Session, slug: str):
    """Get a prank by any of its slugs, raising 404 if not found."""
    from ..models import Prank
    prank = (
        db.query(Prank)
        .filter((Prank.share_slug == slug) | (Prank.builder_slug == slug) | (Prank.slug == slug))
        .first()
    )
    if prank is None:
        raise HTTPException(status_code=404, detail="Prank not found")
    return prank


def build_text_request(
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


def routing_metadata(decision: RoutingDecision) -> RoutingMetadata:
    return RoutingMetadata(
        chosen_model_id=decision.chosen_model_id,
        scores=decision.scores,
        tags=decision.tags,
        reason=decision.reason,
    )


def sleep_prank_delay() -> None:
    """Artificial delay for prank matches to mimic generation time."""
    if PRANK_DELAY_MS > 0:
        time.sleep(PRANK_DELAY_MS / 1000.0)


# ---------------------------------------------------------------------------
# Client IP
# ---------------------------------------------------------------------------


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"
