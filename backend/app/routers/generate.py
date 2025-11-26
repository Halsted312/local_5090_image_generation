"""Image generation and related endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Header, Query, Response, WebSocket
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from starlette.websockets import WebSocketDisconnect

from ..database import get_db
from ..metrics import get_percentiles
from ..model_registry import MODEL_REGISTRY
from ..models import GenerationLog
from ..queue_manager import QueueFull
from ..schemas import GenerationLogEntry, ImageResponse, TextGenerateRequest
from ..services.deps import (
    GEN_QUEUE,
    PERCENTILE_CACHE_TTL_SECONDS,
    file_media_type,
    tf32_enabled,
)
from ..services.generation import process_generation, record_metric

logger = logging.getLogger(__name__)

router = APIRouter(tags=["generate"])


@router.options("/api/generate")
async def generate_options() -> Response:
    """Explicit preflight handler; CORSMiddleware adds headers."""
    return Response(status_code=200)


@router.get("/api/queue/status")
def get_queue_status() -> dict:
    """Return global queue status for synthetic generation coordination."""
    queue_length = len(GEN_QUEUE._queue)
    return {
        "queue_length": queue_length,
        "is_idle": queue_length == 0,
    }


@router.post("/api/generate", response_model=ImageResponse)
def generate_image(
    request: TextGenerateRequest,
    x_benchmark_run: str | None = Header(None, alias="X-Benchmark-Run"),
    db: Session = Depends(get_db),
) -> ImageResponse:
    session_id = request.session_id or "anon"
    try:
        job = GEN_QUEUE.enqueue(session_id)
    except QueueFull:
        raise HTTPException(status_code=429, detail="Queue is full, please retry shortly")

    # Block until this job reaches the front of the queue.
    GEN_QUEUE.wait_for_turn(job["generation_id"])
    queue_wait_ms = int((time.time() - job["enqueued_at"]) * 1000)
    queue_position = job.get("queue_position")

    try:
        processing_started_at = datetime.now(timezone.utc)
        resp = process_generation(request, db)
        processing_finished_at = datetime.now(timezone.utc)
        GEN_QUEUE.release(
            job["generation_id"],
            success=True,
            payload={
                "image_base64": resp.image_base64,
                "model_used": resp.model_used or resp.model_id,
            },
        )
        resp.queue_wait_ms = queue_wait_ms
        resp.processing_time_ms = resp.generation_time_ms
        resp.prompt = request.prompt
        resp.image_url = resp.image_path

        model_for_distribution = resp.model_used or resp.model_id
        # Persist metrics (skip if model missing)
        if model_for_distribution:
            duration_ms = resp.generation_time_ms or int(
                (processing_finished_at - processing_started_at).total_seconds() * 1000
            )
            is_synthetic = (x_benchmark_run or "").lower() in {"1", "true", "yes"}
            try:
                record_metric(
                    db,
                    prompt=request.prompt,
                    model_used=model_for_distribution,
                    engine_requested=request.engine,
                    num_steps=request.num_inference_steps,
                    guidance=request.guidance_scale,
                    width=request.width,
                    height=request.height,
                    seed=request.seed,
                    tf32_enabled=tf32_enabled(),
                    is_synthetic=is_synthetic,
                    is_prank=False,
                    queue_position=queue_position,
                    queue_wait_ms=queue_wait_ms,
                    duration_ms=duration_ms,
                    started_at=processing_started_at,
                    ended_at=processing_finished_at,
                    router_json=resp.router_metadata.model_dump() if resp.router_metadata else None,
                    session_id=request.session_id,
                    share_slug=None,
                    prompt_metadata=request.benchmark_meta,
                )
            except Exception as exc:
                logger.warning("Failed to record generation metric: %s", exc)
        # Attach percentile distributions (per model and all models) after recording
        try:
            resp.distribution, _ = get_percentiles(db, model_for_distribution, cache_ttl_seconds=PERCENTILE_CACHE_TTL_SECONDS)
            resp.distribution_all, _ = get_percentiles(db, "all", cache_ttl_seconds=PERCENTILE_CACHE_TTL_SECONDS)
        except Exception as exc:
            logger.warning("Failed to compute distributions: %s", exc)
        return resp
    except Exception as exc:
        GEN_QUEUE.release(
            job["generation_id"],
            success=False,
            payload={"error": str(exc)},
        )
        raise


@router.get("/api/generations", response_model=list[GenerationLogEntry])
def list_generations(
    limit: int = 50,
    offset: int = 0,
    session_id: str | None = Query(None, alias="sessionId"),
    db: Session = Depends(get_db),
) -> list[GenerationLogEntry]:
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    # Privacy: require session_id to filter; otherwise return empty list.
    if not session_id:
        return []

    rows = (
        db.query(GenerationLog)
        .filter(GenerationLog.session_id == session_id)
        .order_by(GenerationLog.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [
        GenerationLogEntry(
            id=str(r.id),
            prompt=r.prompt,
            model_id=r.model_id,
            image_path=r.image_path,
            thumbnail_path=r.thumbnail_path,
            created_at=r.created_at.isoformat(),
            share_slug=r.share_slug,
            router_json=json.loads(r.router_json) if r.router_json else None,
            session_id=r.session_id,
        )
        for r in rows
    ]


@router.get("/api/metrics/distribution")
def metrics_distribution(
    model_id: str = Query("all", alias="modelId"),
    db: Session = Depends(get_db),
):
    """
    Return latency percentiles (p5..p100) for the given model over the last 30 days.
    Cached for ~5 minutes to avoid heavy DB scans.
    """
    model_key = None if model_id in (None, "", "all") else model_id
    distribution, count = get_percentiles(db, model_key, cache_ttl_seconds=PERCENTILE_CACHE_TTL_SECONDS)
    return {
        "modelId": model_id if model_key else "all",
        "distribution": distribution,
        "count": count,
        "asOf": datetime.now(timezone.utc).isoformat(),
    }


@router.websocket("/ws/queue")
async def queue_updates(websocket: WebSocket, sessionId: str | None = Query(None, alias="sessionId")):
    """
    WebSocket endpoint to stream queue updates for a session.
    """
    session_id = sessionId or "anon"
    await websocket.accept()
    GEN_QUEUE.hub.set_loop(asyncio.get_running_loop())
    GEN_QUEUE.hub.register(session_id, websocket)
    try:
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
    finally:
        GEN_QUEUE.hub.unregister(session_id, websocket)


@router.get("/api/images/thumb/{generation_id}")
def get_generation_thumbnail(
    generation_id: str,
    db: Session = Depends(get_db),
):
    gen = db.query(GenerationLog).filter(GenerationLog.id == generation_id).first()
    if not gen or not gen.thumbnail_path:
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    thumb_path = Path(gen.thumbnail_path)
    if not thumb_path.is_absolute():
        thumb_path = (Path(gen.thumbnail_path)).resolve()
    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail file missing")

    return FileResponse(thumb_path, media_type=file_media_type(thumb_path))


@router.get("/api/models")
def list_models_api() -> dict:
    """
    Expose model registry for frontend to consume.
    """
    models = []
    for model_id, info in MODEL_REGISTRY.items():
        models.append(
            {
                "id": model_id,
                "display_name": info.get("display_name", model_id),
                "tags": info.get("tags", []),
                "notes": info.get("notes", ""),
            }
        )
    return {"models": models}
