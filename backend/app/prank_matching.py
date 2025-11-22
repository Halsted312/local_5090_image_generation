"""Heuristic + optional LLM-based prank trigger matching."""

from __future__ import annotations

import json
import logging
import os
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


# Thresholds for deciding when to rely solely on heuristics vs calling the LLM.
HIGH_CONFIDENCE_THRESHOLD = 0.95
MIN_LLM_THRESHOLD = 0.25


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
                        torch_dtype=torch.float32,
                        device_map={"": "cpu"},
                    )
                    self._pipe = pipeline(
                        "text-generation",
                        model=mdl,
                        tokenizer=tok,
                        device=-1,  # CPU
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

    def choose_with_candidates(
        self,
        prompt: str,
        candidate_texts: List[str],
        candidate_scores: List[float],
    ) -> Optional[int]:
        """
        Ask the LLM to pick among a small set of candidate triggers.
        Returns index into candidate_texts (0..len-1) or None.
        """
        if not candidate_texts:
            return None

        system = (
            "You are a prank trigger matcher. "
            "Given a user prompt and a list of candidate triggers with heuristic scores, "
            "pick the SINGLE best candidate index or -1 if none match. "
            'Respond ONLY with a JSON object: {"index": <int>} and nothing else.'
        )

        numbered = []
        for i, (text, score) in enumerate(zip(candidate_texts, candidate_scores)):
            numbered.append(f"{i}: {text!r} (heuristic_score={score:.3f})")
        numbered_str = "\n".join(numbered)

        user = f"User prompt: {prompt!r}\nCandidates:\n{numbered_str}\n"
        full_prompt = f"{system}\n\n{user}\nAnswer:"

        try:
            pipe = self._get_pipe()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load prank matcher LLM: %s", exc)
            return None

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
            if isinstance(idx, int) and 0 <= idx < len(candidate_texts):
                return idx
        except Exception as exc:  # noqa: BLE001
            logger.warning("Prank matcher LLM parse failed: %s | raw: %r", exc, out[:200])
            return None

        return None


# ---------------------------------------------------------------------------
# Global loader
# ---------------------------------------------------------------------------

PRANK_MATCHER_LLM: PrankMatcherLLM | None = None
_GLOBAL_LLM_LOCK = threading.Lock()


def get_prank_matcher_llm() -> PrankMatcherLLM | None:
    """
    Lazily load a single global PrankMatcherLLM instance.
    Returns None when PRANK_MATCHER_LLM_ID is not configured.
    """
    global PRANK_MATCHER_LLM
    model_id = os.getenv("PRANK_MATCHER_LLM_ID")
    if not model_id:
        return None
    if PRANK_MATCHER_LLM is None:
        with _GLOBAL_LLM_LOCK:
            if PRANK_MATCHER_LLM is None:
                PRANK_MATCHER_LLM = PrankMatcherLLM(model_id)
    return PRANK_MATCHER_LLM


# ---------------------------------------------------------------------------
# Unified match function
# ---------------------------------------------------------------------------


def match_prank_trigger(
    prompt: str,
    triggers: List[str],
    llm: Optional[PrankMatcherLLM] = None,
    top_k: int = 2,
) -> Tuple[Optional[int], MatchDebug]:
    """
    Use heuristics to score triggers; delegate to LLM to pick among top candidates when needed.
    """
    heuristic_idx, scores = heuristic_match(prompt, triggers)

    # No triggers at all
    if not triggers:
        debug = MatchDebug(
            prompt=prompt,
            triggers=triggers,
            heuristic_idx=None,
            heuristic_scores=[],
            used_llm=False,
            llm_idx=None,
            final_idx=None,
        )
        logger.info("Prank match debug: %s", debug)
        return None, debug

    ranked = sorted(enumerate(scores), key=lambda kv: kv[1], reverse=True)
    best_idx, best_score = ranked[0]

    # If heuristics are extremely confident, short-circuit.
    if best_score >= HIGH_CONFIDENCE_THRESHOLD:
        final_idx = best_idx
        debug = MatchDebug(
            prompt=prompt,
            triggers=triggers,
            heuristic_idx=heuristic_idx,
            heuristic_scores=scores,
            used_llm=False,
            llm_idx=None,
            final_idx=final_idx,
        )
        logger.info("Prank match debug (heuristics only): %s", debug)
        return final_idx, debug

    used_llm = False
    llm_global_idx: Optional[int] = None

    # Only consider LLM if we have a configured model and a minimally reasonable heuristic score.
    if llm is not None and best_score >= MIN_LLM_THRESHOLD:
        used_llm = True
        candidates = ranked[: max(1, min(top_k, len(ranked)))]
        candidate_indices = [i for (i, _) in candidates]
        candidate_texts = [triggers[i] for i in candidate_indices]
        candidate_scores = [scores[i] for i in candidate_indices]

        llm_local_idx = llm.choose_with_candidates(prompt, candidate_texts, candidate_scores)
        if llm_local_idx is not None and 0 <= llm_local_idx < len(candidate_indices):
            llm_global_idx = candidate_indices[llm_local_idx]

    # Determine final index with fallbacks.
    if llm_global_idx is not None:
        final_idx = llm_global_idx
    elif best_score >= MIN_LLM_THRESHOLD:
        final_idx = best_idx
    else:
        final_idx = None

    debug = MatchDebug(
        prompt=prompt,
        triggers=triggers,
        heuristic_idx=heuristic_idx,
        heuristic_scores=scores,
        used_llm=used_llm,
        llm_idx=llm_global_idx,
        final_idx=final_idx,
    )
    logger.info("Prank match debug: %s", debug)
    return final_idx, debug
