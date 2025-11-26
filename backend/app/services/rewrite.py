"""LLM-based prompt rewriting for NSFW content."""

from __future__ import annotations

import logging
import os
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

REWRITE_MODEL_ID = os.getenv("REWRITE_LLM_ID", os.getenv("ROUTER_LLM_ID", "Qwen/Qwen2.5-1.5B-Instruct"))
_rewrite_pipeline = None
_rewrite_lock = threading.Lock()


def get_rewrite_pipeline():
    """Load or return the small LLM used to clean NSFW prompts."""
    global _rewrite_pipeline
    if _rewrite_pipeline is None:
        with _rewrite_lock:
            if _rewrite_pipeline is None:
                logger.info("Loading rewrite LLM: %s", REWRITE_MODEL_ID)
                _rewrite_pipeline = pipeline(
                    "text-generation",
                    model=AutoModelForCausalLM.from_pretrained(
                        REWRITE_MODEL_ID,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                    ),
                    tokenizer=AutoTokenizer.from_pretrained(REWRITE_MODEL_ID),
                )
    return _rewrite_pipeline


def rewrite_prompt_safe(prompt: str) -> str | None:
    """
    Ask small LLM to make prompt non-NSFW while preserving meaning.
    Returns rewritten prompt or None on failure.
    """
    system = (
        "The prompt below was flagged as NSFW by an image model, but the intent is safe. "
        "Rewrite it to be clearly safe-for-work while preserving meaning. "
        "Disambiguate any words that might imply drugs or explicit content "
        "(e.g., 'weed' -> 'grass', 'burning grass' -> 'controlled burn of vegetation'). "
        "Keep it concise; do not add new concepts."
    )
    try:
        pipe = get_rewrite_pipeline()
    except Exception as exc:
        logger.warning("Rewrite LLM unavailable: %s", exc)
        return None
    try:
        outputs = pipe(
            f"{system}\nPrompt: {prompt}\nRewritten:",
            max_new_tokens=128,
            do_sample=False,
            temperature=0.3,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        text = outputs[0]["generated_text"]
        # Heuristic: take the substring after 'Rewritten:' if present
        marker = "Rewritten:"
        if marker in text:
            return text.split(marker, 1)[1].strip()
        return text.strip()
    except Exception as exc:
        logger.warning("Rewrite LLM call failed: %s", exc)
        return None
