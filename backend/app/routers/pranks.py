"""Prank management and generation endpoints."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from sqlalchemy.orm import Session

from ..database import get_db
from ..metrics import get_percentiles
from ..models import Prank, PrankTrigger
from ..prank_matching import get_prank_matcher_llm, match_prank_trigger
from ..queue_manager import QueueFull
from ..router_engine import RoutingDecision
from ..schemas import (
    ImageResponse,
    MatchedTrigger,
    PrankCreateResponse,
    PrankDetailResponse,
    PrankGenerateRequest,
    PrankMetadataCreate,
    PrankSummary,
    PrankTriggerCreateResponse,
    PrankTriggerUpdateRequest,
)
from ..services.deps import (
    ADMIN_TOKEN_TTL_SECONDS,
    GEN_QUEUE,
    PERCENTILE_CACHE_TTL_SECONDS,
    build_text_request,
    encode_thumbnail_base64,
    file_media_type,
    generate_unique_slug,
    get_admin_token,
    get_admin_token_issued,
    get_prank_by_slug,
    routing_metadata,
    sleep_prank_delay,
    tf32_enabled,
)
from ..services.generation import log_generation, process_generation, record_metric
from ..storage import (
    PRANK_IMAGE_ROOT,
    load_prank_image_base64,
    resolve_image_path,
    save_generation_image,
    save_prank_image_with_thumbnail,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["pranks"])


def _prank_to_response(prank: Prank, triggers: list) -> dict:
    base_url = os.getenv("FRONTEND_BASE_URL", "https://promptpics.ai")
    share_slug = prank.share_slug
    builder_slug = prank.builder_slug

    # Special URL for VIP prank
    if share_slug == "imagine":
        share_url = f"{base_url}/imagine"
    else:
        share_url = f"{base_url}/p/{share_slug}"

    return {
        "id": str(prank.id),
        "slug": share_slug,  # legacy field
        "shareSlug": share_slug,  # explicit shareSlug field
        "builderSlug": builder_slug,
        "title": prank.title,
        "sessionId": prank.session_id,
        "shareUrl": share_url,
        "builderUrl": f"{base_url}/customize/{builder_slug}",
        "createdAt": prank.created_at.isoformat() if prank.created_at else None,
        "viewCount": prank.view_count or 0,
        "isVip": prank.is_vip if hasattr(prank, 'is_vip') else False,
        "triggers": [
            {
                "id": str(t.id),
                "triggerText": t.trigger_text,
                "imageBase64": load_prank_image_base64(t.image_path),
                "thumbnailBase64": load_prank_image_base64(t.thumbnail_path) if t.thumbnail_path else None,
                "createdAt": t.created_at.isoformat() if t.created_at else None,
                "matchCount": t.match_count or 0,
            }
            for t in triggers
        ],
    }


def _prank_to_summary(prank: Prank, trigger_count: int) -> dict:
    base_url = os.getenv("FRONTEND_BASE_URL", "https://promptpics.ai")
    share_slug = prank.share_slug
    builder_slug = prank.builder_slug

    # Special URL for VIP prank
    if share_slug == "imagine":
        share_url = f"{base_url}/imagine"
    else:
        share_url = f"{base_url}/p/{share_slug}"

    return {
        "id": str(prank.id),
        "slug": share_slug,  # legacy field
        "shareSlug": share_slug,  # explicit shareSlug field
        "builderSlug": builder_slug,
        "title": prank.title,
        "sessionId": prank.session_id,
        "shareUrl": share_url,
        "builderUrl": f"{base_url}/customize/{builder_slug}",
        "createdAt": prank.created_at.isoformat() if prank.created_at else None,
        "viewCount": prank.view_count or 0,
        "triggerCount": trigger_count,
        "isVip": prank.is_vip if hasattr(prank, 'is_vip') else False,
    }


def _validate_image_path(slug: str, relative_path: str) -> str:
    resolved = resolve_image_path(slug, relative_path)
    root = PRANK_IMAGE_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid image path") from exc

    if not resolved.exists():
        raise HTTPException(status_code=400, detail=f"Image not found at {relative_path}")
    return str(resolved)


def _add_prank_trigger(
    prank: Prank,
    trigger_text: str,
    file: UploadFile,
    db: Session,
) -> PrankTriggerCreateResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    payload = file.file.read()
    ext = ".png"
    if file.filename and "." in file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower()

    # Use share_slug for storage folder consistency
    image_path, thumb_path = save_prank_image_with_thumbnail(
        prank.share_slug or prank.slug or "prank", payload, extension=ext
    )

    trigger = PrankTrigger(
        prank_id=prank.id,
        trigger_text=trigger_text,
        image_path=image_path,
        thumbnail_path=thumb_path,
    )
    db.add(trigger)
    db.commit()
    db.refresh(trigger)

    return PrankTriggerCreateResponse(
        id=str(trigger.id),
        trigger_text=trigger.trigger_text,
        image_path=trigger.image_path,
        thumbnail_path=trigger.thumbnail_path,
    )


def _check_prank_access(prank: Prank, session_id: str | None, x_admin_token: str | None) -> None:
    """Check if the user has access to modify this prank."""
    # Check if prank is admin-only
    if hasattr(prank, 'is_admin_only') and prank.is_admin_only:
        # Admin-only pranks require valid admin token
        if not x_admin_token or x_admin_token != get_admin_token():
            raise HTTPException(status_code=403, detail="Admin access required for this prank.")
        if (time.time() - get_admin_token_issued()) > ADMIN_TOKEN_TTL_SECONDS:
            raise HTTPException(status_code=401, detail="Admin token expired; please login again")
    else:
        # Regular prank: enforce ownership when session_id is provided
        if prank.session_id and session_id and prank.session_id != session_id:
            raise HTTPException(status_code=403, detail="Not authorized to modify this prank.")


# ---------------------------------------------------------------------------
# Prank CRUD Endpoints
# ---------------------------------------------------------------------------


@router.post("/api/pranks", response_model=PrankCreateResponse)
def create_prank(
    metadata: PrankMetadataCreate,
    db: Session = Depends(get_db),
) -> PrankCreateResponse:
    share_slug = generate_unique_slug(db, length=6)
    builder_slug = generate_unique_slug(db, length=8)
    prank = Prank(
        share_slug=share_slug,
        builder_slug=builder_slug,
        slug=share_slug,  # legacy compatibility
        title=metadata.title,
        session_id=metadata.session_id,
    )
    db.add(prank)
    db.commit()
    db.refresh(prank)

    payload = _prank_to_response(prank, [])
    return PrankCreateResponse(**payload)


# IMPORTANT: /api/pranks/summaries MUST be defined BEFORE /api/pranks/{slug}
# to avoid route collision (FastAPI matches routes in order)
@router.get("/api/pranks/summaries", response_model=list[PrankSummary])
def list_prank_summaries(
    session_id: str | None = Query(None, alias="sessionId"),
    db: Session = Depends(get_db),
) -> list[PrankSummary]:
    """
    Lightweight prank summaries without base64 payloads.
    Must be defined BEFORE /api/pranks/{slug} to avoid route collision.
    """
    if not session_id:
        return []

    pranks = (
        db.query(Prank)
        .filter(Prank.session_id == session_id)
        .order_by(Prank.created_at.desc())
        .all()
    )

    results: list[PrankSummary] = []
    for prank in pranks:
        trigger_count = (
            db.query(PrankTrigger)
            .filter(PrankTrigger.prank_id == prank.id)
            .count()
        )
        payload = _prank_to_summary(prank, trigger_count)
        results.append(PrankSummary(**payload))
    return results


@router.get("/api/pranks/{slug}", response_model=PrankDetailResponse)
def get_prank(slug: str, db: Session = Depends(get_db)) -> PrankDetailResponse:
    prank = get_prank_by_slug(db, slug)

    triggers = (
        db.query(PrankTrigger)
        .filter(PrankTrigger.prank_id == prank.id)
        .order_by(PrankTrigger.created_at.asc())
        .all()
    )
    payload = _prank_to_response(prank, triggers)
    return PrankDetailResponse(**payload)


@router.get("/api/pranks/slug/{slug}", response_model=PrankDetailResponse)
def get_prank_alias(slug: str, db: Session = Depends(get_db)) -> PrankDetailResponse:
    """Alias for GET /api/pranks/{slug} to match alternate caller expectations."""
    return get_prank(slug, db)


@router.get("/api/pranks", response_model=list[PrankDetailResponse])
def list_pranks(
    session_id: str | None = Query(None, alias="sessionId"),
    db: Session = Depends(get_db),
) -> list[PrankDetailResponse]:
    """
    List prank sets owned by the provided session_id. Returns [] when session_id is missing.
    """
    if not session_id:
        return []

    pranks = (
        db.query(Prank)
        .filter(Prank.session_id == session_id)
        .order_by(Prank.created_at.desc())
        .all()
    )

    results: list[PrankDetailResponse] = []
    for prank in pranks:
        triggers = (
            db.query(PrankTrigger)
            .filter(PrankTrigger.prank_id == prank.id)
            .order_by(PrankTrigger.created_at.asc())
            .all()
        )
        payload = _prank_to_response(prank, triggers)
        results.append(PrankDetailResponse(**payload))
    return results


# ---------------------------------------------------------------------------
# Trigger Endpoints
# ---------------------------------------------------------------------------


@router.post("/api/pranks/{slug}/triggers", response_model=PrankTriggerCreateResponse)
def add_prank_trigger_by_slug(
    slug: str,
    session_id: str | None = Form(None, alias="sessionId"),
    trigger_text: str = Form(..., alias="triggerText"),
    file: UploadFile = File(...),
    x_admin_token: str | None = Header(None, alias="X-Admin-Token"),
    db: Session = Depends(get_db),
) -> PrankTriggerCreateResponse:
    prank = db.query(Prank).filter(
        (Prank.share_slug == slug) | (Prank.builder_slug == slug) | (Prank.slug == slug)
    ).first()
    if prank is None:
        raise HTTPException(status_code=404, detail="Prank not found.")

    _check_prank_access(prank, session_id, x_admin_token)
    return _add_prank_trigger(prank, trigger_text, file, db)


@router.patch("/api/pranks/{prank_id}/triggers/{trigger_id}", response_model=PrankTriggerCreateResponse)
def update_prank_trigger(
    prank_id: str,
    trigger_id: str,
    payload: PrankTriggerUpdateRequest,
    db: Session = Depends(get_db),
) -> PrankTriggerCreateResponse:
    trigger = (
        db.query(PrankTrigger)
        .filter(PrankTrigger.id == trigger_id, PrankTrigger.prank_id == prank_id)
        .first()
    )
    if not trigger:
        raise HTTPException(status_code=404, detail="Trigger not found")

    trigger.trigger_text = payload.trigger_text
    db.commit()
    db.refresh(trigger)
    return PrankTriggerCreateResponse(
        id=str(trigger.id),
        trigger_text=trigger.trigger_text,
        image_path=trigger.image_path,
        thumbnail_path=trigger.thumbnail_path,
    )


@router.patch("/api/pranks/slug/{slug}/triggers/{trigger_id}", response_model=PrankTriggerCreateResponse)
def update_prank_trigger_by_slug(
    slug: str,
    trigger_id: str,
    payload: PrankTriggerUpdateRequest,
    session_id: str | None = Query(None, alias="sessionId"),
    x_admin_token: str | None = Header(None, alias="X-Admin-Token"),
    db: Session = Depends(get_db),
) -> PrankTriggerCreateResponse:
    prank = get_prank_by_slug(db, slug)
    _check_prank_access(prank, session_id, x_admin_token)

    trigger = (
        db.query(PrankTrigger)
        .filter(PrankTrigger.id == trigger_id, PrankTrigger.prank_id == prank.id)
        .first()
    )
    if not trigger:
        raise HTTPException(status_code=404, detail="Trigger not found")

    trigger.trigger_text = payload.trigger_text
    db.commit()
    db.refresh(trigger)
    return PrankTriggerCreateResponse(
        id=str(trigger.id),
        trigger_text=trigger.trigger_text,
        image_path=trigger.image_path,
        thumbnail_path=trigger.thumbnail_path,
    )


@router.delete("/api/pranks/{prank_id}/triggers/{trigger_id}")
def delete_prank_trigger(
    prank_id: str,
    trigger_id: str,
    db: Session = Depends(get_db),
) -> dict:
    trigger = (
        db.query(PrankTrigger)
        .filter(PrankTrigger.id == trigger_id, PrankTrigger.prank_id == prank_id)
        .first()
    )
    if not trigger:
        raise HTTPException(status_code=404, detail="Trigger not found")

    db.delete(trigger)
    db.commit()
    return {"detail": "Trigger deleted"}


@router.delete("/api/pranks/slug/{slug}/triggers/{trigger_id}")
def delete_prank_trigger_by_slug(
    slug: str,
    trigger_id: str,
    session_id: str | None = Query(None, alias="sessionId"),
    x_admin_token: str | None = Header(None, alias="X-Admin-Token"),
    db: Session = Depends(get_db),
) -> dict:
    prank = get_prank_by_slug(db, slug)
    _check_prank_access(prank, session_id, x_admin_token)

    trigger = (
        db.query(PrankTrigger)
        .filter(PrankTrigger.id == trigger_id, PrankTrigger.prank_id == prank.id)
        .first()
    )
    if not trigger:
        raise HTTPException(status_code=404, detail="Trigger not found")

    db.delete(trigger)
    db.commit()
    return {"detail": "Trigger deleted"}


@router.post("/api/pranks/{slug}/triggers/{trigger_id}/edit", response_model=PrankTriggerCreateResponse)
def edit_prank_trigger_image(
    slug: str,
    trigger_id: str,
    session_id: str | None = Form(None, alias="sessionId"),
    image: UploadFile = File(...),
    edit_description: str | None = Form(None, alias="edit_description"),
    x_admin_token: str | None = Header(None, alias="X-Admin-Token"),
    db: Session = Depends(get_db),
) -> PrankTriggerCreateResponse:
    prank = get_prank_by_slug(db, slug)
    _check_prank_access(prank, session_id, x_admin_token)

    trigger = (
        db.query(PrankTrigger)
        .filter(PrankTrigger.id == trigger_id, PrankTrigger.prank_id == prank.id)
        .first()
    )
    if not trigger:
        raise HTTPException(status_code=404, detail="Trigger not found")

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    payload = image.file.read()
    ext = ".png"
    if image.filename and "." in image.filename:
        ext = "." + image.filename.rsplit(".", 1)[-1].lower()

    image_path, thumb_path = save_prank_image_with_thumbnail(
        prank.share_slug or prank.slug or "prank",
        payload,
        extension=ext,
    )

    trigger.image_path = image_path
    trigger.thumbnail_path = thumb_path
    db.commit()
    db.refresh(trigger)

    return PrankTriggerCreateResponse(
        id=str(trigger.id),
        trigger_text=trigger.trigger_text,
        image_path=trigger.image_path,
        thumbnail_path=trigger.thumbnail_path,
    )


@router.get("/api/pranks/{slug}/triggers/{trigger_id}/thumbnail")
def get_prank_trigger_thumbnail(
    slug: str,
    trigger_id: str,
    db: Session = Depends(get_db),
):
    prank = get_prank_by_slug(db, slug)
    trigger = (
        db.query(PrankTrigger)
        .filter(PrankTrigger.id == trigger_id, PrankTrigger.prank_id == prank.id)
        .first()
    )
    if not trigger:
        raise HTTPException(status_code=404, detail="Trigger not found")

    thumb_path = trigger.thumbnail_path or trigger.image_path
    if not thumb_path:
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    path = Path(thumb_path)
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail file missing")

    return FileResponse(path, media_type=file_media_type(path))


# ---------------------------------------------------------------------------
# Prank Generation Endpoint
# ---------------------------------------------------------------------------


@router.post("/api/p/{slug}/generate", response_model=ImageResponse)
def generate_prank_image(
    slug: str,
    request: PrankGenerateRequest,
    x_benchmark_run: str | None = Header(None, alias="X-Benchmark-Run"),
    db: Session = Depends(get_db),
) -> ImageResponse:
    session_id = request.session_id or "anon"
    try:
        job = GEN_QUEUE.enqueue(session_id)
    except QueueFull:
        raise HTTPException(status_code=429, detail="Queue is full, please retry shortly")

    GEN_QUEUE.wait_for_turn(job["generation_id"])
    queue_wait_ms = int((time.time() - job["enqueued_at"]) * 1000)
    queue_position = job.get("queue_position")

    try:
        prank = get_prank_by_slug(db, slug)

        triggers = db.query(PrankTrigger).filter(PrankTrigger.prank_id == prank.id).all()
        prank.view_count = (prank.view_count or 0) + 1
        db.commit()

        text_request = build_text_request(
            prompt=request.prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            seed=request.seed,
            engine=request.engine or "auto",
            session_id=request.session_id,
        )
        if not triggers:
            processing_started_at = datetime.now(timezone.utc)
            resp = process_generation(text_request, db, share_slug=prank.share_slug)
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
            if model_for_distribution and model_for_distribution != "prank":
                duration_ms = resp.generation_time_ms or int(
                    (processing_finished_at - processing_started_at).total_seconds() * 1000
                )
                is_synthetic = (x_benchmark_run or "").lower() in {"1", "true", "yes"}
                try:
                    record_metric(
                        db,
                        prompt=request.prompt,
                        model_used=model_for_distribution,
                        engine_requested=request.engine or "auto",
                        num_steps=text_request.num_inference_steps,
                        guidance=text_request.guidance_scale,
                        width=text_request.width,
                        height=text_request.height,
                        seed=text_request.seed,
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
                        share_slug=prank.share_slug,
                        prompt_metadata=None,
                    )
                except Exception as exc:
                    logger.warning("Failed to record generation metric (prank no-trigger): %s", exc)
            try:
                resp.distribution, _ = get_percentiles(db, model_for_distribution, cache_ttl_seconds=PERCENTILE_CACHE_TTL_SECONDS)
                resp.distribution_all, _ = get_percentiles(db, "all", cache_ttl_seconds=PERCENTILE_CACHE_TTL_SECONDS)
            except Exception as exc:
                logger.warning("Failed to compute distributions: %s", exc)
            return resp

        trap_texts = [t.trigger_text for t in triggers]
        # Heuristic + optional LLM matcher
        matcher_llm = get_prank_matcher_llm()
        idx, debug = match_prank_trigger(request.prompt, trap_texts, llm=matcher_llm)
        start_time = time.time()
        logger.info(
            "Prank match debug for slug=%s prompt=%r: %s",
            slug,
            request.prompt,
            debug,
        )

        if idx is not None:
            trigger = triggers[idx]
            # Save a copy + thumbnail into generation storage for logging consistency.
            try:
                with Image.open(trigger.image_path) as prank_img:
                    image_path, thumb_path = save_generation_image(prank_img)
            except Exception as exc:
                logger.exception("Failed to load prank image for slug %s", slug)
                raise HTTPException(status_code=500, detail="Failed to load prank image") from exc

            trigger.match_count = (trigger.match_count or 0) + 1
            db.commit()

            decision = RoutingDecision(
                chosen_model_id="prank",
                scores={"prank": 1.0},
                tags=["prank"],
                reason=f"Matched prank trigger: {trigger.trigger_text}",
            )
            generation_id = log_generation(
                db=db,
                prompt=request.prompt,
                model_id="prank",
                router_decision=decision,
                image_path=image_path,
                thumbnail_path=thumb_path,
                prank_id=str(prank.id),
                share_slug=prank.share_slug,
                session_id=request.session_id,
            )

            # Artificial delay to mimic generation time for prank hits.
            sleep_prank_delay()
            total_ms = int((time.time() - start_time) * 1000)

            resp = ImageResponse(
                image_base64=load_prank_image_base64(trigger.image_path),
                generation_id=generation_id,
                model_id="prank",
                model_used="prank",
                thumbnail_base64=encode_thumbnail_base64(thumb_path),
                image_path=image_path,
                thumbnail_path=thumb_path,
                router_metadata=routing_metadata(decision),
                was_prank=True,
                matched_trigger_id=str(trigger.id),
                matched_trigger_text=trigger.trigger_text,
                generation_time_ms=total_ms,
                is_prank_match=True,
                matched_trigger=MatchedTrigger(
                    id=str(trigger.id),
                    trigger_text=trigger.trigger_text,
                    image_base64=load_prank_image_base64(trigger.image_path),
                    thumbnail_base64=encode_thumbnail_base64(trigger.thumbnail_path)
                    if trigger.thumbnail_path
                    else None,
                ),
            )
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
            return resp

        processing_started_at = datetime.now(timezone.utc)
        resp = process_generation(text_request, db, share_slug=prank.share_slug)
        processing_finished_at = datetime.now(timezone.utc)
        # Explicitly mark as non-prank when no trigger matched.
        resp.is_prank_match = False
        resp.matched_trigger = None
        resp.matched_trigger_id = None
        resp.matched_trigger_text = None
        resp.was_prank = False
        if resp.model_id and not resp.model_used:
            resp.model_used = resp.model_id
        resp.queue_wait_ms = queue_wait_ms
        resp.processing_time_ms = resp.generation_time_ms
        resp.prompt = request.prompt
        resp.image_url = resp.image_path
        model_for_distribution = resp.model_used or resp.model_id
        if model_for_distribution and model_for_distribution != "prank":
            duration_ms = resp.generation_time_ms or int(
                (processing_finished_at - processing_started_at).total_seconds() * 1000
            )
            is_synthetic = (x_benchmark_run or "").lower() in {"1", "true", "yes"}
            try:
                record_metric(
                    db,
                    prompt=request.prompt,
                    model_used=model_for_distribution,
                    engine_requested=request.engine or "auto",
                    num_steps=text_request.num_inference_steps,
                    guidance=text_request.guidance_scale,
                    width=text_request.width,
                    height=text_request.height,
                    seed=text_request.seed,
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
                    share_slug=prank.share_slug,
                    prompt_metadata=None,
                )
            except Exception as exc:
                logger.warning("Failed to record generation metric (prank no match): %s", exc)
        try:
            resp.distribution, _ = get_percentiles(db, model_for_distribution, cache_ttl_seconds=PERCENTILE_CACHE_TTL_SECONDS)
            resp.distribution_all, _ = get_percentiles(db, "all", cache_ttl_seconds=PERCENTILE_CACHE_TTL_SECONDS)
        except Exception as exc:
            logger.warning("Failed to compute distributions: %s", exc)
        GEN_QUEUE.release(
            job["generation_id"],
            success=True,
            payload={
                "image_base64": resp.image_base64,
                "model_used": resp.model_used or resp.model_id,
            },
        )
        return resp
    except Exception as exc:
        GEN_QUEUE.release(
            job["generation_id"],
            success=False,
            payload={"error": str(exc)},
        )
        raise
