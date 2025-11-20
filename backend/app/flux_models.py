"""Lazy-loaded FLUX pipelines used by the FastAPI endpoints."""

from __future__ import annotations

import logging
import threading

import torch
from diffusers import FluxKontextPipeline, FluxPipeline

from .config import FLUX_KONTEXT_MODEL_ID, FLUX_TEXT_MODEL_ID, get_device

logger = logging.getLogger(__name__)

_text_pipeline: FluxPipeline | None = None
_kontext_pipeline: FluxKontextPipeline | None = None
_lock = threading.Lock()


def _load_text_pipeline() -> FluxPipeline:
    """Instantiate the text-to-image FLUX pipeline."""
    device = get_device()
    logger.info("Loading FLUX text-to-image pipeline (%s) on %s", FLUX_TEXT_MODEL_ID, device)
    try:
        pipeline = FluxPipeline.from_pretrained(
            FLUX_TEXT_MODEL_ID,
            torch_dtype=torch.bfloat16,
        )
        pipeline.to(device)
        # Uncomment if you need to conserve VRAM.
        # pipeline.enable_model_cpu_offload()
    except Exception as exc:
        logger.exception("Failed to initialize FLUX text pipeline")
        raise RuntimeError(f"Failed to load FLUX text pipeline: {exc}") from exc
    return pipeline


def _load_kontext_pipeline() -> FluxKontextPipeline:
    """Instantiate the image-editing FLUX Kontext pipeline."""
    device = get_device()
    logger.info("Loading FLUX Kontext pipeline (%s) on %s", FLUX_KONTEXT_MODEL_ID, device)
    try:
        pipeline = FluxKontextPipeline.from_pretrained(
            FLUX_KONTEXT_MODEL_ID,
            torch_dtype=torch.bfloat16,
        )
        pipeline.to(device)
        # Uncomment if you need to conserve VRAM.
        # pipeline.enable_model_cpu_offload()
    except Exception as exc:
        logger.exception("Failed to initialize FLUX Kontext pipeline")
        raise RuntimeError(f"Failed to load FLUX Kontext pipeline: {exc}") from exc
    return pipeline


def get_text_pipeline() -> FluxPipeline:
    """
    Return a singleton FLUX text-to-image pipeline.

    Pipelines are heavy to load, so we instantiate them once per process.
    """
    global _text_pipeline
    if _text_pipeline is None:
        with _lock:
            if _text_pipeline is None:
                _text_pipeline = _load_text_pipeline()
    return _text_pipeline


def get_kontext_pipeline() -> FluxKontextPipeline:
    """
    Return a singleton FLUX Kontext (image editing) pipeline.

    Pipelines are heavy to load, so we instantiate them once per process.
    """
    global _kontext_pipeline
    if _kontext_pipeline is None:
        with _lock:
            if _kontext_pipeline is None:
                _kontext_pipeline = _load_kontext_pipeline()
    return _kontext_pipeline
