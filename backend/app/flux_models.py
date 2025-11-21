"""Lazy-loaded FLUX text-to-image pipeline used by FastAPI."""

from __future__ import annotations

import logging
import threading
import os

import torch
from diffusers import FluxPipeline

from .config import FLUX_TEXT_MODEL_ID, get_device

logger = logging.getLogger(__name__)

_text_pipeline: FluxPipeline | None = None
_lock = threading.Lock()


def _load_text_pipeline() -> FluxPipeline:
    """Instantiate the text-to-image FLUX pipeline."""
    device = get_device()
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    logger.info("Loading FLUX text-to-image pipeline (%s) on %s", FLUX_TEXT_MODEL_ID, device)
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
        pipeline = FluxPipeline.from_pretrained(
            FLUX_TEXT_MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=token,
        )
        try:
            pipeline.to(device)
        except torch.cuda.OutOfMemoryError:
            logger.warning("Text pipeline OOM on GPU; enabling CPU offload.")
            pipeline.enable_model_cpu_offload()
        pipeline.enable_attention_slicing()
        pipeline.enable_vae_slicing()
    except Exception as exc:
        logger.exception("Failed to initialize FLUX text pipeline")
        raise RuntimeError(f"Failed to load FLUX text pipeline: {exc}") from exc
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
