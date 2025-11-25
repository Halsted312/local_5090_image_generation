"""Model routing logic (heuristics + optional LLM fallback).

This implements the explicit scoring rules from docs/1_model_selection.md.
It exposes:
  - score_models_for_prompt(): fast, deterministic heuristic router.
  - route_prompt(): uses heuristics, and if confidence is low, calls a small LLM
    (Meta Llama 3 8B Instruct by default) to return a JSON RoutingDecision.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


ImageModelId = str  # Expected: "flux_dev", "realvis_xl", "sd3_medium", "hidream_dev"
EXCLUDED_MODELS: set[str] = set(
    m.strip()
    for m in os.getenv("EXCLUDED_LIVE_MODELS", "").split(",")
    if m.strip()
)


@dataclass
class RoutingDecision:
    """Container for routing output."""

    chosen_model_id: ImageModelId
    scores: Dict[ImageModelId, float]
    tags: List[str] = field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# Heuristic rules (from docs/1_model_selection.md)
# ---------------------------------------------------------------------------

PORTRAIT_WORDS = [
    "portrait",
    "selfie",
    "headshot",
    "close-up",
    "close up",
    "bust shot",
    "studio photo",
    "studio portrait",
    "fashion shoot",
    "beauty shot",
    "model pose",
    "profile photo",
    "id photo",
    "passport photo",
    "face of",
    "photo of a man",
    "photo of a woman",
    "realistic man",
    "realistic woman",
    "realistic person",
]

HUMAN_WORDS = [
    "man",
    "woman",
    "boy",
    "girl",
    "person",
    "people",
    "child",
    "children",
    "teen",
    "adult",
    "old man",
    "old woman",
    "elderly",
]

PHOTO_WORDS = [
    "photo",
    "photograph",
    "photography",
    "dslr",
    "35mm",
    "canon",
    "nikon",
    "f/1.8",
    "f/2.8",
    "bokeh",
    "raw photo",
    "raw photograph",
    "high resolution photo",
    "8k photo",
    "real life",
    "realistic",
]

LOGO_WORDS = [
    "logo",
    "icon",
    "badge",
    "emblem",
    "crest",
    "monogram",
    "shield logo",
    "wordmark",
    "brand mark",
    "app icon",
    "favicon",
    "business card logo",
    "company logo",
    "branding",
]

TEXT_LAYOUT_WORDS = [
    "typography",
    "headline",
    "title text",
    "big title",
    "poster with text",
    "flyer",
    "brochure",
    "magazine cover",
    "book cover",
    "movie poster",
    "album cover",
    "banner",
    "billboard",
    "sign",
    "signage",
    "label",
    "package design",
    "infographic",
]

UI_LAYOUT_WORDS = [
    "ui",
    "user interface",
    "dashboard",
    "web app ui",
    "landing page",
    "mobile app screen",
    "wireframe",
    "mockup",
    "website design",
]

STYLE_CARTOON_WORDS = [
    "cartoon",
    "anime",
    "manga",
    "pixar style",
    "3d render",
    "illustration",
    "flat illustration",
    "vector art",
    "clip art",
]

LANDSCAPE_WORDS = [
    "landscape",
    "scenery",
    "mountain",
    "forest",
    "lake",
    "river",
    "ocean",
    "beach",
    "cityscape",
    "skyline",
    "sunset",
    "sunrise",
    "valley",
    "desert",
    "countryside",
    "fields",
]


def _contains_any(prompt: str, word_list: List[str]) -> bool:
    p = prompt.lower()
    return any(w in p for w in word_list)


def _count_multi_clause_markers(prompt: str) -> int:
    p = prompt.lower()
    markers = [" and ", " with ", " while ", " in the background", " beside ", " behind "]
    return sum(p.count(m) for m in markers)


def score_models_for_prompt(
    prompt: str,
) -> Tuple[ImageModelId, Dict[ImageModelId, float], List[str], str]:
    # Force flux_dev for all prompts for now.
    return "flux_dev", {"flux_dev": 1.0}, ["forced_flux"], "Forced flux_dev routing"


# ---------------------------------------------------------------------------
# LLM fallback router
# ---------------------------------------------------------------------------

ROUTER_MODEL_ID = os.getenv("ROUTER_LLM_ID", "Qwen/Qwen2.5-1.5B-Instruct")
_router_pipeline = None
_router_lock = threading.Lock()

_SYSTEM_PROMPT = """
You are an "image model router". Decide which ONE engine to use:
- flux_dev: general-purpose, strong prompt following.
- realvis_xl: photorealistic faces/people.
- sd3_medium: complex prompts, text inside image, posters/UI.
- hidream_dev: HiDream I1 Dev - text rendering, logos/wordmarks with readable text.

Rules:
1) If human/portrait + photo language -> realvis_xl >= 0.85, flux_dev secondary.
2) If logo/icon/branding/wordmark -> hidream_dev >= 0.8; especially if text must be readable.
3) If posters/UI/text layout -> sd3_medium >= 0.8; hidream_dev if text clarity critical.
4) If long/complex multi-clause -> boost sd3_medium and flux_dev.
5) Stylized/cartoon -> flux_dev highest.
6) Landscapes without portraits/logos -> flux_dev preferred.
7) Tie-break: portraits -> realvis_xl; logos/text -> hidream_dev; complex -> sd3_medium; else flux_dev.

Output ONLY JSON:
{
  "chosen_model_id": "<flux_dev|realvis_xl|sd3_medium|hidream_dev>",
  "scores": {"flux_dev":0.0,"realvis_xl":0.0,"sd3_medium":0.0,"hidream_dev":0.0},
  "tags": ["portrait","logo",...],
  "reason": "short explanation"
}
Scores between 0 and 1 and sum ~1.
""".strip()


def _get_router_pipeline():
    global _router_pipeline
    if _router_pipeline is None:
        with _router_lock:
            if _router_pipeline is None:
                logger.info("Loading router LLM: %s", ROUTER_MODEL_ID)
                _router_pipeline = pipeline(
                    "text-generation",
                    model=AutoModelForCausalLM.from_pretrained(
                        ROUTER_MODEL_ID,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                    ),
                    tokenizer=AutoTokenizer.from_pretrained(ROUTER_MODEL_ID),
                )
    return _router_pipeline


def _build_router_prompt(user_prompt: str) -> str:
    return f"{_SYSTEM_PROMPT}\n\nUSER PROMPT:\n{user_prompt.strip()}\n"


def _build_router_prompt_with_hints(
    user_prompt: str, heuristic_scores: Dict[str, float], heuristic_tags: List[str]
) -> str:
    hints = json.dumps(
        {
            "heuristic_scores": heuristic_scores,
            "heuristic_tags": heuristic_tags,
        },
        ensure_ascii=False,
    )
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Heuristic analysis:\n{hints}\n\n"
        f"USER PROMPT:\n{user_prompt.strip()}\n"
    )


def _extract_json_block(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def route_with_llm(
    prompt: str, heuristic_scores: Dict[str, float] | None = None, heuristic_tags: List[str] | None = None
) -> RoutingDecision | None:
    """Call small LLM router to get a structured decision."""
    try:
        pipe = _get_router_pipeline()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Router LLM unavailable, falling back to heuristics: %s", exc)
        return None

    if heuristic_scores is not None or heuristic_tags is not None:
        prompt_text = _build_router_prompt_with_hints(
            prompt, heuristic_scores or {}, heuristic_tags or []
        )
    else:
        prompt_text = _build_router_prompt(prompt)
    try:
        outputs = pipe(
            prompt_text,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.1,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        text = outputs[0]["generated_text"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Router LLM call failed, falling back to heuristics: %s", exc)
        return None

    data = _extract_json_block(text)
    if not data:
        logger.warning("Router LLM returned non-JSON: %r", text[:200])
        return None
    try:
        chosen = data["chosen_model_id"]
        scores = data.get("scores") or {}
        tags = data.get("tags") or []
        reason = data.get("reason") or ""
        return RoutingDecision(chosen_model_id=chosen, scores=scores, tags=tags, reason=reason)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse router JSON: %s", exc)
        return None


def route_prompt(prompt: str, confidence_gap: float = 0.2, min_confidence: float = 0.7) -> RoutingDecision:
    """
    Route using heuristics; if low confidence, fall back to LLM.

    Args:
        prompt: user text prompt.
        confidence_gap: required gap between top2 to skip LLM.
        min_confidence: required best score to skip LLM.
    """
    best_id, scores, tags, reason = score_models_for_prompt(prompt)
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_score = sorted_scores[0][1]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

    if best_score >= min_confidence and (best_score - second_score) >= confidence_gap:
        return RoutingDecision(chosen_model_id=best_id, scores=scores, tags=tags, reason=f"Heuristic: {reason}")

    try:
        llm_decision = route_with_llm(prompt, heuristic_scores=scores, heuristic_tags=tags)
        if llm_decision:
            if llm_decision.chosen_model_id in EXCLUDED_MODELS:
                # Fallback to best non-excluded heuristic
                filtered = [(mid, sc) for mid, sc in scores.items() if mid not in EXCLUDED_MODELS]
                filtered = sorted(filtered, key=lambda kv: kv[1], reverse=True)
                fallback_id = filtered[0][0] if filtered else best_id
                return RoutingDecision(
                    chosen_model_id=fallback_id,
                    scores=scores,
                    tags=tags,
                    reason=f"LLM chose excluded model; fallback to heuristic {fallback_id}",
                )
            return llm_decision
    except Exception as exc:  # noqa: BLE001
        logger.warning("Routing LLM failure, using heuristic fallback: %s", exc)

    # Fallback to heuristic if LLM fails
    return RoutingDecision(chosen_model_id=best_id, scores=scores, tags=tags, reason=f"Heuristic (fallback): {reason}")
