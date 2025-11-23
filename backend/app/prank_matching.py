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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import AutoModel, AutoTokenizer as EmbeddingTokenizer

from .config import (
    PRANK_EMBED_MODEL_ID,
    PRANK_EMBED_ACCEPT_THRESHOLD,
    PRANK_EMBED_REJECT_THRESHOLD,
    PRANK_MAX_LEN_RATIO,
    PRANK_DISQUALIFY_TERMS,
)
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
MIN_LLM_THRESHOLD = 0.25  # legacy LLM path (still used as a fallback)


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
    """Legacy heuristic placeholder; replaced by embedding similarity."""
    scores: List[float] = [0.0] * len(triggers)
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
            "Given a user prompt and a list of candidate triggers with similarity scores, "
            "pick the SINGLE best candidate index ONLY if the prompt describes the SAME SCENARIO as that trigger, "
            "without adding new actions/objects/people/locations. If none match, return -1. "
            'Respond ONLY with a JSON object: {"index": <int>} and nothing else.'
        )

        numbered = []
        for i, (text, score) in enumerate(zip(candidate_texts, candidate_scores)):
            numbered.append(f"{i}: {text!r} (sim_score={score:.3f})")
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

# Embedding model globals
_EMBED_MODEL = None
_EMBED_TOKENIZER = None
_EMBED_LOCK = threading.Lock()
_EMBED_CACHE: Dict[str, torch.Tensor] = {}
_EMBED_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def _load_embedder():
    global _EMBED_MODEL, _EMBED_TOKENIZER
    if _EMBED_MODEL is None or _EMBED_TOKENIZER is None:
        with _EMBED_LOCK:
            if _EMBED_MODEL is None or _EMBED_TOKENIZER is None:
                logger.info("Loading prank embedding model: %s on %s", PRANK_EMBED_MODEL_ID, _EMBED_DEVICE)
                _EMBED_TOKENIZER = EmbeddingTokenizer.from_pretrained(PRANK_EMBED_MODEL_ID)
                _EMBED_MODEL = AutoModel.from_pretrained(PRANK_EMBED_MODEL_ID).to(_EMBED_DEVICE)


def _embed(text: str) -> torch.Tensor:
    text = _normalize(text)
    if text in _EMBED_CACHE:
        return _EMBED_CACHE[text]
    _load_embedder()
    assert _EMBED_MODEL is not None and _EMBED_TOKENIZER is not None
    with torch.no_grad():
        tokens = _EMBED_TOKENIZER(text, return_tensors="pt", truncation=True).to(_EMBED_DEVICE)
        model_out = _EMBED_MODEL(**tokens)
        embeddings = model_out.last_hidden_state
        # Mean pooling
        attention_mask = tokens["attention_mask"].unsqueeze(-1)
        summed = torch.sum(embeddings * attention_mask, dim=1)
        counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        pooled = summed / counts
        pooled = F.normalize(pooled, p=2, dim=1)
        vec = pooled[0].detach().cpu()
        _EMBED_CACHE[text] = vec
        return vec


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
    Use embeddings + optional LLM confirmation to pick the best trigger.
    - Normalize inputs
    - Compute cosine similarities
    - Disqualify when prompt adds violent/sexual/action terms not in trigger
    - Accept if high cosine AND no disqualifiers
    - Otherwise ask LLM to confirm “same scenario”
    """
    prompt_norm = _normalize(prompt)
    if not triggers:
        debug = MatchDebug(prompt=prompt, triggers=triggers, heuristic_idx=None, heuristic_scores=[], used_llm=False, llm_idx=None, final_idx=None)
        logger.info("Prank match debug: %s", debug)
        return None, debug

    prompt_tokens = set(_tokenize(prompt_norm))
    trigger_tokens = [_tokenize(_normalize(t)) for t in triggers]
    trig_norms = [_normalize(t) for t in triggers]

    # Disqualifier helper: if prompt adds disallowed terms not present in trigger
    def _has_disqualifier(idx: int) -> bool:
        trig_set = set(trigger_tokens[idx])
        new_terms = prompt_tokens - trig_set
        for term in PRANK_DISQUALIFY_TERMS:
            if term in new_terms:
                return True
        return False

    # Compute embeddings
    prompt_vec = _embed(prompt_norm)
    trig_vecs = [_embed(tn) for tn in trig_norms]
    scores = []
    for i, vec in enumerate(trig_vecs):
        sim = float(torch.matmul(prompt_vec, vec))
        if _has_disqualifier(i):
            sim = 0.0
        scores.append(sim)

    ranked = sorted(enumerate(scores), key=lambda kv: kv[1], reverse=True)
    best_idx, best_score = ranked[0]

    # Quick accept if very close and lengths are comparable
    prompt_len = max(len(prompt_tokens), 1)
    trig_len = max(len(trigger_tokens[best_idx]), 1)
    len_ratio = prompt_len / trig_len

    final_idx: Optional[int] = None
    used_llm = False
    llm_idx: Optional[int] = None

    if best_score >= PRANK_EMBED_ACCEPT_THRESHOLD and len_ratio <= PRANK_MAX_LEN_RATIO:
        final_idx = best_idx
    elif best_score < PRANK_EMBED_REJECT_THRESHOLD:
        final_idx = None
    else:
        # Borderline: ask LLM to confirm “same scenario”
        if llm is not None:
            used_llm = True
            candidates = ranked[: max(1, min(top_k, len(ranked)))]
            candidate_indices = [i for (i, _) in candidates]
            candidate_texts = [trig_norms[i] for i in candidate_indices]
            candidate_scores = [scores[i] for i in candidate_indices]

            # Use existing chooser but prompt is stricter in the class.
            llm_local_idx = llm.choose_with_candidates(prompt_norm, candidate_texts, candidate_scores)
            if llm_local_idx is not None and 0 <= llm_local_idx < len(candidate_indices):
                llm_idx = candidate_indices[llm_local_idx]
                final_idx = llm_idx
        # If LLM not available or declined, reject unless score is very high and length ratio okay
        if final_idx is None and best_score >= PRANK_EMBED_ACCEPT_THRESHOLD and len_ratio <= PRANK_MAX_LEN_RATIO:
            final_idx = best_idx

    debug = MatchDebug(
        prompt=prompt,
        triggers=triggers,
        heuristic_idx=None,
        heuristic_scores=scores,
        used_llm=used_llm,
        llm_idx=llm_idx,
        final_idx=final_idx,
    )
    logger.info("Prank match debug: %s", debug)
    return final_idx, debug
