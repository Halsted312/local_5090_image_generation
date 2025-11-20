"""Configuration helpers for FLUX model selection and device choice."""

from __future__ import annotations

import logging
import os
from typing import Literal

import torch

logger = logging.getLogger(__name__)

# Default model IDs; can be overridden via environment variables.
FLUX_TEXT_MODEL_ID: str = os.getenv("FLUX_TEXT_MODEL_ID", "black-forest-labs/FLUX.1-schnell")
FLUX_KONTEXT_MODEL_ID: str = os.getenv("FLUX_KONTEXT_MODEL_ID", "black-forest-labs/FLUX.1-Kontext-dev")

DeviceType = Literal["cuda", "cpu"]


def get_device() -> DeviceType:
    """
    Choose the best available device for inference.

    Prefers CUDA when available, otherwise falls back to CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    logger.warning("CUDA not available, falling back to CPU. Performance will degrade.")
    return "cpu"
