"""Lazy-loaded FLUX text-to-image pipeline used by FastAPI."""

from __future__ import annotations

import logging
import threading
import os

import torch
from diffusers import FluxPipeline, StableDiffusion3Pipeline, StableDiffusionXLPipeline

from .config import (
    FLUX_TEXT_MODEL_ID,
    LOGO_SDXL_MODEL_ID,
    REALVIS_MODEL_ID,
    SD3_MODEL_ID,
    get_device,
)

logger = logging.getLogger(__name__)

_text_pipeline: FluxPipeline | None = None
_realvis_pipeline: StableDiffusionXLPipeline | None = None
_sd3_pipeline: StableDiffusion3Pipeline | None = None
_logo_pipeline: StableDiffusionXLPipeline | None = None
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


def _load_realvis_pipeline() -> StableDiffusionXLPipeline:
    device = get_device()
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    logger.info("Loading RealVisXL pipeline (%s) on %s", REALVIS_MODEL_ID, device)
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
        pipe = StableDiffusionXLPipeline.from_pretrained(
            REALVIS_MODEL_ID, torch_dtype=torch.bfloat16, token=token
        )
        try:
            pipe.to(device)
        except torch.cuda.OutOfMemoryError:
            logger.warning("RealVis pipeline OOM on GPU; enabling CPU offload.")
            pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize RealVis pipeline")
        raise RuntimeError(f"Failed to load RealVis pipeline: {exc}") from exc
    return pipe


def get_realvis_pipeline() -> StableDiffusionXLPipeline:
    global _realvis_pipeline
    if _realvis_pipeline is None:
        with _lock:
            if _realvis_pipeline is None:
                _realvis_pipeline = _load_realvis_pipeline()
    return _realvis_pipeline


def _load_sd3_pipeline() -> StableDiffusion3Pipeline:
    device = get_device()
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    logger.info("Loading SD3-Medium pipeline (%s) on %s", SD3_MODEL_ID, device)
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
        pipe = StableDiffusion3Pipeline.from_pretrained(
            SD3_MODEL_ID, torch_dtype=torch.bfloat16, token=token
        )
        try:
            pipe.to(device)
        except torch.cuda.OutOfMemoryError:
            logger.warning("SD3 pipeline OOM on GPU; enabling CPU offload.")
            pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize SD3 pipeline")
        raise RuntimeError(f"Failed to load SD3 pipeline: {exc}") from exc
    return pipe


def get_sd3_pipeline() -> StableDiffusion3Pipeline:
    global _sd3_pipeline
    if _sd3_pipeline is None:
        with _lock:
            if _sd3_pipeline is None:
                _sd3_pipeline = _load_sd3_pipeline()
    return _sd3_pipeline


def _load_logo_pipeline() -> StableDiffusionXLPipeline:
    device = get_device()
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    logger.info("Loading SDXL logo pipeline (%s) on %s", LOGO_SDXL_MODEL_ID, device)
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
        pipe = StableDiffusionXLPipeline.from_pretrained(
            LOGO_SDXL_MODEL_ID, torch_dtype=torch.bfloat16, token=token
        )
        try:
            pipe.to(device)
        except torch.cuda.OutOfMemoryError:
            logger.warning("Logo SDXL pipeline OOM on GPU; enabling CPU offload.")
            pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize logo SDXL pipeline")
        raise RuntimeError(f"Failed to load logo SDXL pipeline: {exc}") from exc
    return pipe


def get_logo_pipeline() -> StableDiffusionXLPipeline:
    global _logo_pipeline
    if _logo_pipeline is None:
        with _lock:
            if _logo_pipeline is None:
                _logo_pipeline = _load_logo_pipeline()
    return _logo_pipeline
