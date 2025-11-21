"""Configuration helpers for FLUX model selection and device choice."""

from __future__ import annotations

import logging
import os
from typing import Literal

import torch

logger = logging.getLogger(__name__)

# Default model ID; can be overridden via environment variables.
FLUX_TEXT_MODEL_ID: str = os.getenv("FLUX_TEXT_MODEL_ID", "black-forest-labs/FLUX.1-schnell")

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
