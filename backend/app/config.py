"""Configuration helpers for FLUX model selection and device choice."""

from __future__ import annotations

import logging
import os
from typing import Literal

import torch

logger = logging.getLogger(__name__)

# Default model IDs; can be overridden via environment variables.
FLUX_TEXT_MODEL_ID: str = os.getenv("FLUX_TEXT_MODEL_ID", "black-forest-labs/FLUX.1-schnell")
REALVIS_MODEL_ID: str = os.getenv("REALVIS_MODEL_ID", "SG161222/RealVisXL_V4.0")
SD3_MODEL_ID: str = os.getenv("SD3_MODEL_ID", "stabilityai/stable-diffusion-3-medium-diffusers")
# HiDream-I1-Full replaces SDXL for superior text/logo generation
LOGO_SDXL_MODEL_ID: str = os.getenv("LOGO_SDXL_MODEL_ID", "HiDream-ai/HiDream-I1-Full")

# HiDream-specific parameters
HIDREAM_STEPS: int = int(os.getenv("HIDREAM_STEPS", "40"))
HIDREAM_GUIDANCE: float = float(os.getenv("HIDREAM_GUIDANCE", "5.0"))
# Llama text encoder for HiDream
HIDREAM_TEXT_ENCODER_ID: str = os.getenv("HIDREAM_TEXT_ENCODER_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")

DeviceType = Literal["cuda", "cpu"]


def get_device() -> DeviceType:
    """
    Choose the best available device for inference.

    Prefers CUDA when available, otherwise falls back to CPU.
    """
    forced = os.getenv("DEVICE") or os.getenv("FORCE_DEVICE")
    if forced and forced.lower() == "cpu":
        logger.warning("Forcing CPU due to DEVICE/FORCE_DEVICE override.")
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    logger.warning("CUDA not available, falling back to CPU. Performance will degrade.")
    return "cpu"
