"""Local disk storage utilities for prank images."""

from __future__ import annotations

import base64
import os
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PRANK_IMAGE_ROOT = Path(os.getenv("PRANK_IMAGE_ROOT", BASE_DIR / "prank_images"))
PRANK_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)


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
