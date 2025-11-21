from __future__ import annotations

import base64
import io
import logging
from typing import Iterable

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .flux_models import get_text_pipeline
from .schemas import ImageResponse, TextGenerateRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FLUX Image API")

ALLOWED_ORIGINS: Iterable[str] = (
    "http://localhost:6970",
    "http://127.0.0.1:6970",
    "http://localhost",
    "http://127.0.0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _pil_to_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _make_generator(device: torch.device | str, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    return torch.Generator(device=str(device)).manual_seed(seed)


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

    return ImageResponse(image_base64=_pil_to_base64_png(image))
