"""Utilities for latency percentile computation with caching."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple, Dict

from sqlalchemy.orm import Session

from .models import GenerationMetric

Percentiles = List[int]

# Cache structure: key -> (timestamp_seconds, distribution, count)
_CACHE: Dict[str, Tuple[float, Percentiles, int]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes
_WINDOW_DAYS = 30


def _nearest_rank_percentile(sorted_values: List[int], percentile: float) -> int:
    """
    Compute percentile using the nearest-rank method.
    percentile expected between 0 and 100.
    """
    if not sorted_values:
        return 0
    n = len(sorted_values)
    rank = max(1, int((percentile / 100.0) * n + 0.9999))
    rank = min(rank, n)
    return sorted_values[rank - 1]


def _compute_distribution(values: List[int]) -> Percentiles:
    buckets = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    values_sorted = sorted(values)
    return [_nearest_rank_percentile(values_sorted, p) for p in buckets]


def _cache_key(model_id: str | None) -> str:
    return model_id or "all"


def get_percentiles(
    db: Session,
    model_id: str | None = None,
    cache_ttl_seconds: int = _CACHE_TTL_SECONDS,
) -> Tuple[Percentiles, int]:
    """
    Return (distribution, count) for the requested model over the last _WINDOW_DAYS.
    Cached for cache_ttl_seconds to reduce DB load.
    """
    key = _cache_key(model_id)
    now = time.time()
    cached = _CACHE.get(key)
    if cached and (now - cached[0]) < cache_ttl_seconds:
        return cached[1], cached[2]

    cutoff = datetime.now(timezone.utc) - timedelta(days=_WINDOW_DAYS)
    q = db.query(GenerationMetric.duration_ms).filter(GenerationMetric.started_at >= cutoff)
    if model_id and model_id != "all":
        q = q.filter(GenerationMetric.model_used == model_id)
    durations = [row[0] for row in q.all() if row[0] is not None]
    distribution = _compute_distribution(durations) if durations else [0] * 20
    _CACHE[key] = (now, distribution, len(durations))
    return distribution, len(durations)


def clear_cache() -> None:
    _CACHE.clear()
