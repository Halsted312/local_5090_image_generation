"""Local disk storage utilities for prank images."""

from __future__ import annotations

import base64
import os
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


def _slug_folder(slug: str) -> Path:
    safe_slug = slug.replace("/", "_")
    folder = PRANK_IMAGE_ROOT / safe_slug
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_prank_image(slug: str, payload: bytes, extension: str = ".png") -> str:
    """
    Save prank image bytes under ./prank_images/<slug>/<uuid>.ext and return the absolute path.
    """
    folder = _slug_folder(slug)
    filename = f"{uuid.uuid4().hex}{extension}"
    path = folder / filename
    with open(path, "wb") as f:
        f.write(payload)
    return str(path.resolve())


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
    filename = f"{uuid.uuid4().hex}{extension}"
    path = (folder / filename).resolve()
    thumb_path = (folder / f"{filename}_thumb.jpg").resolve()

    with open(path, "wb") as f:
        f.write(payload)

    # build thumbnail
    with Image.open(path) as img:
        thumb = img.convert("RGB")
        thumb.thumbnail((256, 256))
        thumb.save(thumb_path, format="JPEG", quality=80)

    return str(path), str(thumb_path)


def save_generation_image(image: Image.Image) -> Tuple[str, str]:
    """
    Save a generated PIL image and a thumbnail.

    Returns:
        (image_path, thumbnail_path) absolute paths.
    """
    image_id = uuid.uuid4().hex
    img_path = (GEN_IMAGE_ROOT / f"{image_id}.png").resolve()
    thumb_path = (GEN_THUMB_ROOT / f"{image_id}_thumb.jpg").resolve()

    img_path.parent.mkdir(parents=True, exist_ok=True)
    thumb_path.parent.mkdir(parents=True, exist_ok=True)

    image.save(img_path, format="PNG")

    thumb = image.copy()
    thumb.thumbnail((256, 256))
    thumb.save(thumb_path, format="JPEG", quality=80)

    return str(img_path), str(thumb_path)
