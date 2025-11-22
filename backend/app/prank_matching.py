"""Heuristic + optional LLM-based prank trigger matching."""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes and helpers
# ---------------------------------------------------------------------------


@dataclass
class MatchDebug:
    prompt: str
    triggers: List[str]
    heuristic_idx: Optional[int]
    heuristic_scores: List[float]
    used_llm: bool
    llm_idx: Optional[int]
    final_idx: Optional[int]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", _normalize(text))


def _jaccard(a_tokens: List[str], b_tokens: List[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union else 0.0


def heuristic_match(prompt: str, triggers: List[str]) -> Tuple[Optional[int], List[float]]:
    """
    Heuristic matching:
    - exact match
    - substring/contains
    - Jaccard token overlap
    Returns (index, scores) where scores[i] is in [0,1].
    """
    prompt_norm = _normalize(prompt)
    prompt_tokens = _tokenize(prompt_norm)

    scores: List[float] = [0.0] * len(triggers)
    best_idx: Optional[int] = None
    best_score = 0.0

    for i, trig in enumerate(triggers):
        trig_norm = _normalize(trig)
        trig_tokens = _tokenize(trig_norm)

        score = 0.0

        # exact
        if prompt_norm == trig_norm:
            score = 1.0
        # substring
        elif trig_norm in prompt_norm or prompt_norm in trig_norm:
            score = 0.9
        else:
            j = _jaccard(prompt_tokens, trig_tokens)
            score = 0.8 * j

        scores[i] = score
        if score > best_score:
            best_score = score
            best_idx = i

    # confidence threshold
    if best_score >= 0.7:
        return best_idx, scores

    return None, scores


# ---------------------------------------------------------------------------
# Optional LLM matcher (lazy-loaded)
# ---------------------------------------------------------------------------


class PrankMatcherLLM:
    """Small LLM classifier that picks best trigger index."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._lock = threading.Lock()
        self._pipe = None

    def _get_pipe(self):
        if self._pipe is None:
            with self._lock:
                if self._pipe is None:
                    logger.info("Loading prank matcher LLM: %s", self.model_id)
                    tok = AutoTokenizer.from_pretrained(self.model_id)
                    mdl = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16,
                        device_map="auto",
                    )
                    self._pipe = pipeline(
                        "text-generation",
                        model=mdl,
                        tokenizer=tok,
                    )
        return self._pipe

    def choose(self, prompt: str, triggers: List[str]) -> Optional[int]:
        if not triggers:
            return None
        try:
            pipe = self._get_pipe()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load prank matcher LLM: %s", exc)
            return None

        system = (
            "You are a prank trigger matcher. "
            "Given a user prompt and a numbered list of trigger phrases, "
            "pick the SINGLE best matching trigger index, or -1 if none match. "
            "Return ONLY a JSON object: {\"index\": <int>} and nothing else."
        )
        numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(triggers))
        user = f"User prompt: {prompt}\nTriggers:\n{numbered}\n"
        full_prompt = f"{system}\n\n{user}\nAnswer:"

        try:
            out = pipe(
                full_prompt,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
            )[0]["generated_text"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Prank matcher LLM call failed: %s", exc)
            return None

        try:
            start = out.find("{")
            end = out.rfind("}") + 1
            blob = out[start:end]
            data = json.loads(blob)
            idx = data.get("index")
            if isinstance(idx, int) and 0 <= idx < len(triggers):
                return idx
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Prank matcher LLM parse failed: %s | raw: %r", exc, out[:200])
            return None


# ---------------------------------------------------------------------------
# Unified match function
# ---------------------------------------------------------------------------


def match_prank_trigger(
    prompt: str,
    triggers: List[str],
    llm: Optional[PrankMatcherLLM] = None,
) -> Tuple[Optional[int], MatchDebug]:
    heuristic_idx, scores = heuristic_match(prompt, triggers)
    used_llm = False
    llm_idx: Optional[int] = None

    if heuristic_idx is None and llm is not None and triggers:
        used_llm = True
        llm_idx = llm.choose(prompt, triggers)

    final_idx = heuristic_idx if heuristic_idx is not None else llm_idx

    debug = MatchDebug(
        prompt=prompt,
        triggers=triggers,
        heuristic_idx=heuristic_idx,
        heuristic_scores=scores,
        used_llm=used_llm,
        llm_idx=llm_idx,
        final_idx=final_idx,
    )
    logger.info("Prank match debug: %s", debug)
    return final_idx, debug
