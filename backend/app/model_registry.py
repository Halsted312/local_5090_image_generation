"""Registry describing supported image models and their roles."""

from __future__ import annotations

from typing import Dict, List

MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    "flux_dev": {
        "display_name": "FLUX.1-dev",
        "tags": ["general", "stylized", "landscape"],
        "notes": "General-purpose, strong prompt following.",
    },
    "realvis_xl": {
        "display_name": "RealVisXL V4.0",
        "tags": ["portrait", "photoreal"],
        "notes": "Photorealistic faces/people.",
    },
    "sd3_medium": {
        "display_name": "Stable Diffusion 3 Medium",
        "tags": ["complex", "text", "ui"],
        "notes": "Complex scenes, posters, UI, better typography.",
    },
    "logo_sdxl": {
        "display_name": "HiDream I1 (Text & Logos)",
        "tags": ["logo", "icon", "text", "prompt-following"],
        "notes": "17B HiDream-I1-Full, superior text rendering and logo generation.",
    },
}


def list_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_id: str) -> Dict[str, object] | None:
    return MODEL_REGISTRY.get(model_id)
