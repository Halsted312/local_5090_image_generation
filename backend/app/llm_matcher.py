"""LLM-based semantic matcher for prank triggers."""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
_MODEL_ID = os.getenv("PRANK_LLM_ID", DEFAULT_MODEL_ID)

_matcher_pipeline = None
_lock = threading.Lock()


def _get_matcher_pipeline():
    global _matcher_pipeline
    if _matcher_pipeline is None:
        with _lock:
            if _matcher_pipeline is None:
                logger.info("Loading prank LLM matcher: %s", _MODEL_ID)
                _matcher_pipeline = pipeline(
                    "text-generation",
                    model=AutoModelForCausalLM.from_pretrained(
                        _MODEL_ID,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                    ),
                    tokenizer=AutoTokenizer.from_pretrained(_MODEL_ID),
                )
    return _matcher_pipeline


_SYSTEM_INSTRUCTIONS = (
    "You are a strict JSON-only classifier.\n"
    "Given a user prompt and a numbered list of trap prompts, decide if the user prompt "
    "has essentially the SAME intent as one of the trap prompts.\n"
    "- Ignore capitalization, punctuation, minor rewordings, and synonyms.\n"
    "- Only match if a human would say they are asking for the same image.\n"
    "- If none match, answer match=false.\n"
    "Return ONLY a JSON object, nothing else, with this exact shape:\n"
    '{"match": true|false, "index": integer or null}\n'
    "Where index is 1-based index of the matching trap prompt, or null.\n"
)


def _build_prompt(user_prompt: str, trap_prompts: List[str]) -> str:
    traps_block = "\n".join(f"{i+1}. {t}" for i, t in enumerate(trap_prompts))
    return (
        f"{_SYSTEM_INSTRUCTIONS}\n\n"
        f"TRAP_PROMPTS:\n{traps_block}\n\n"
        f"USER_PROMPT:\n{user_prompt}\n\n"
        "Now respond with the JSON object."
    )


def _extract_json(text: str) -> Optional[dict]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def choose_matching_trigger(user_prompt: str, trap_prompts: List[str]) -> Optional[int]:
    """
    Returns the 0-based index of the matching trap prompt, or None.
    """
    if not trap_prompts:
        return None

    pipe = _get_matcher_pipeline()
    prompt = _build_prompt(user_prompt, trap_prompts)

    outputs = pipe(
        prompt,
        max_new_tokens=64,
        do_sample=False,
        temperature=0.1,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )
    text = outputs[0]["generated_text"]
    data = _extract_json(text)
    if not data:
        logger.warning("Matcher LLM returned non-JSON: %r", text[:200])
        return None

    if not isinstance(data, dict) or not data.get("match"):
        return None

    idx = data.get("index")
    try:
        idx_int = int(idx)
    except (TypeError, ValueError):
        return None

    if 1 <= idx_int <= len(trap_prompts):
        return idx_int - 1
    return None
