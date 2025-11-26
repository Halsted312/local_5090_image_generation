"""Local disk storage utilities for prank images."""

from __future__ import annotations

import base64
import os
from io import BytesIO
import uuid
from pathlib import Path
from typing import Tuple

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
PRANK_IMAGE_ROOT = Path(os.getenv("PRANK_IMAGE_ROOT", BASE_DIR / "prank_images"))
PRANK_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

# Generation storage (AI outputs)
GEN_IMAGE_ROOT = Path(os.getenv("GEN_IMAGE_ROOT", BASE_DIR / "data" / "images" / "generations"))
GEN_THUMB_ROOT = Path(os.getenv("GEN_THUMB_ROOT", BASE_DIR / "data" / "images" / "thumbnails"))
GEN_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
GEN_THUMB_ROOT.mkdir(parents=True, exist_ok=True)

# Bench outputs
BENCH_IMAGE_ROOT = Path(os.getenv("BENCH_IMAGE_ROOT", BASE_DIR / "data" / "images" / "bench"))
BENCH_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "512") or "512")
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(1 * 1024 * 1024)) or str(1 * 1024 * 1024))  # 1MB
JPEG_QUALITY_START = 90
JPEG_QUALITY_MIN = 70


def _slug_folder(slug: str) -> Path:
    safe_slug = slug.replace("/", "_")
    folder = PRANK_IMAGE_ROOT / safe_slug
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _normalize_image(image: Image.Image) -> Image.Image:
    """
    Convert to RGB and clamp longest side to MAX_IMAGE_DIM, preserving aspect.
    """
    img = image.convert("RGB")
    max_dim = max(img.size)
    if max_dim > MAX_IMAGE_DIM:
        img = img.copy()
        img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM))
    return img


def _save_jpeg_with_limit(image: Image.Image, path: Path, quality: int = JPEG_QUALITY_START) -> int:
    """
    Save image as JPEG, reducing quality in steps until under MAX_IMAGE_BYTES or reaching min quality.
    Returns the number of bytes written.
    """
    q = quality
    while True:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=q)
        data = buffer.getvalue()
        if len(data) <= MAX_IMAGE_BYTES or q <= JPEG_QUALITY_MIN:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
            return len(data)
        q = max(JPEG_QUALITY_MIN, q - 10)


def save_prank_image(slug: str, payload: bytes, extension: str = ".png") -> str:
    """
    Save prank image bytes under ./prank_images/<slug>/<uuid>.jpg and return the absolute path.
    """
    folder = _slug_folder(slug)
    filename = f"{uuid.uuid4().hex}.jpg"
    path = (folder / filename).resolve()

    with Image.open(BytesIO(payload)) as img:
        normalized = _normalize_image(img)
        _save_jpeg_with_limit(normalized, path, quality=JPEG_QUALITY_START)
    return str(path)


def resolve_image_path(slug: str, relative_path: str) -> Path:
    """
    Resolve the configured prank image root, scoped by slug, with a path provided by the operator.
    """
    safe_slug = slug.replace("/", "_")
    return (PRANK_IMAGE_ROOT / safe_slug / relative_path).resolve()


def load_prank_image_base64(path: str) -> str:
    """
    Read an image from disk and return base64-encoded content.
    """
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


def save_prank_image_with_thumbnail(slug: str, payload: bytes, extension: str = ".png") -> tuple[str, str]:
    """
    Save prank image and a 256px thumbnail. Returns (image_path, thumb_path).
    """
    folder = _slug_folder(slug)
    filename = f"{uuid.uuid4().hex}.jpg"
    path = (folder / filename).resolve()
    thumb_path = (folder / f"{filename}_thumb.jpg").resolve()

    with Image.open(BytesIO(payload)) as img:
        normalized = _normalize_image(img)
        _save_jpeg_with_limit(normalized, path, quality=JPEG_QUALITY_START)

        thumb = normalized.copy()
        thumb.thumbnail((256, 256))
        _save_jpeg_with_limit(thumb, thumb_path, quality=80)

    return str(path), str(thumb_path)


def save_generation_image(image: Image.Image) -> Tuple[str, str]:
    """
    Save a generated PIL image and a thumbnail.

    Returns:
        (image_path, thumbnail_path) absolute paths.
    """
    image_id = uuid.uuid4().hex
    img_path = (GEN_IMAGE_ROOT / f"{image_id}.jpg").resolve()
    thumb_path = (GEN_THUMB_ROOT / f"{image_id}_thumb.jpg").resolve()

    img_path.parent.mkdir(parents=True, exist_ok=True)
    thumb_path.parent.mkdir(parents=True, exist_ok=True)

    normalized = _normalize_image(image)
    _save_jpeg_with_limit(normalized, img_path, quality=JPEG_QUALITY_START)

    thumb = normalized.copy()
    thumb.thumbnail((256, 256))
    _save_jpeg_with_limit(thumb, thumb_path, quality=80)

    return str(img_path), str(thumb_path)


def save_bench_image(run_id: str, engine: str, image: Image.Image) -> str:
    """
    Save a bench PNG image and return the absolute path.
    """
    filename = f"{run_id}_{engine}_{uuid.uuid4().hex}.png"
    img_path = (BENCH_IMAGE_ROOT / filename).resolve()
    img_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(img_path, format="PNG")
    return str(img_path)
