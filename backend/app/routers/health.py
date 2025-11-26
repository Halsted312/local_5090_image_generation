"""Health check and GPU status endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ..services.deps import GPU_COORD_GRACE_SECONDS, GPU_COORD_PATH, read_gpu_coord_owner

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/gpu-status")
def gpu_status() -> dict:
    """Lightweight coordination endpoint for benchmark runner."""
    owner, since = read_gpu_coord_owner()
    busy = owner is not None
    return {
        "busy": busy,
        "owner": owner,
        "since": since,
        "grace_seconds": GPU_COORD_GRACE_SECONDS,
        "path": str(GPU_COORD_PATH) if GPU_COORD_PATH else None,
    }
