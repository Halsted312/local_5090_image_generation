"""Image generation service with model execution and logging."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone

from fastapi import HTTPException
from PIL import Image
from sqlalchemy.orm import Session

from ..flux_models import (
    get_logo_pipeline,
    get_realvis_pipeline,
    get_sd3_pipeline,
    get_text_pipeline,
)
from ..models import GenerationLog, GenerationMetric
from ..router_engine import RoutingDecision, route_prompt
from ..schemas import ImageResponse, TextGenerateRequest
from ..storage import save_generation_image
from ..metrics import clear_cache
from .deps import (
    MAX_IMAGE_SIDE,
    MODEL_PRESETS,
    acquire_gpu_coord,
    encode_thumbnail_base64,
    free_cuda_memory,
    is_black_image,
    make_generator,
    pil_to_base64_png,
    release_gpu_coord,
    routing_metadata,
    wait_for_gpu,
)
from .rewrite import rewrite_prompt_safe

logger = logging.getLogger(__name__)

# Generation lock - initialized by main app
_generation_lock: threading.Lock | None = None


def set_generation_lock(lock: threading.Lock) -> None:
    """Set the generation lock from the main app."""
    global _generation_lock
    _generation_lock = lock


def get_generation_lock() -> threading.Lock:
    """Get the generation lock, raising if not initialized."""
    if _generation_lock is None:
        raise RuntimeError("Generation lock not initialized")
    return _generation_lock


def execute_model(model_id: str, request: TextGenerateRequest) -> tuple[Image.Image, str]:
    """Execute the selected model, branching to the appropriate pipeline."""
    # Clear any lingering CUDA allocations before selecting a pipeline.
    free_cuda_memory()
    selected = model_id
    generation_lock = get_generation_lock()

    if model_id == "flux_dev":
        pipe = get_text_pipeline()
    elif model_id == "realvis_xl":
        pipe = get_realvis_pipeline()
    elif model_id == "sd3_medium":
        pipe = get_sd3_pipeline()
    elif model_id == "hidream_dev":
        pipe = get_logo_pipeline()
    else:
        logger.warning("Model %s not implemented; falling back to flux_dev", model_id)
        pipe = get_text_pipeline()
        selected = "flux_dev"

    # Use HiDream-specific defaults when logo_sdxl is selected
    if model_id == "hidream_dev":
        from ..config import HIDREAM_STEPS, HIDREAM_GUIDANCE
        num_steps = request.num_inference_steps or HIDREAM_STEPS
        guidance = request.guidance_scale if request.guidance_scale is not None else HIDREAM_GUIDANCE
    else:
        num_steps = request.num_inference_steps
        guidance = request.guidance_scale

    # Get device, handling special cases for CPU offload models
    device = getattr(pipe, "device", "cpu")
    # When using device_map or CPU offload, device might be "meta" - use "cpu" for generator
    if str(device) == "meta":
        generator_device = "cpu"
    else:
        generator_device = device
    generator = make_generator(generator_device, request.seed)

    with generation_lock:
        wait_for_gpu(owner="backend")
        acquire_gpu_coord(owner="backend")
        try:
            result = pipe(
                prompt=request.prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                width=request.width,
                height=request.height,
                generator=generator,
            )
            image = result.images[0]
            nsfw = getattr(result, "nsfw_content_detected", None)
            if nsfw is not None:
                logger.info("NSFW check for model %s prompt %r: %s", selected, request.prompt, nsfw)
            # If flagged NSFW, attempt to rewrite prompt and retry once.
            nsfw_flag = False
            if isinstance(nsfw, list):
                nsfw_flag = any(nsfw)
            elif isinstance(nsfw, bool):
                nsfw_flag = nsfw
            if nsfw_flag or is_black_image(image):
                rewritten = rewrite_prompt_safe(request.prompt)
                if rewritten and rewritten != request.prompt:
                    logger.info("Retrying generation for %s with rewritten safe prompt: %r -> %r", selected, request.prompt, rewritten)
                    result = pipe(
                        prompt=rewritten,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance,
                        width=request.width,
                        height=request.height,
                        generator=make_generator(generator_device, request.seed),
                    )
                    image = result.images[0]
                    nsfw_retry = getattr(result, "nsfw_content_detected", None)
                    if nsfw_retry is not None:
                        logger.info("NSFW check after rewrite for %s: %s", selected, nsfw_retry)
                    if is_black_image(image):
                        logger.warning("Image still black after rewrite for %s", selected)
                        if model_id == "logo_sdxl":
                            # Last-resort fallback to flux_dev to avoid returning black image.
                            pipe = get_text_pipeline()
                            selected = "flux_dev"
                            fallback_device = getattr(pipe, "device", "cpu")
                            fallback_generator = make_generator(fallback_device if str(fallback_device) != "meta" else "cpu", request.seed)
                            result = pipe(
                                prompt=rewritten,
                                num_inference_steps=request.num_inference_steps,
                                guidance_scale=request.guidance_scale,
                                width=request.width,
                                height=request.height,
                                generator=fallback_generator,
                            )
                            image = result.images[0]
        except Exception as exc:
            # If HiDream/logo path fails, fall back to flux_dev to return something instead of 500.
            if model_id == "hidream_dev":
                logger.exception("Logo/HiDream generation failed, falling back to flux_dev")
                try:
                    pipe = get_text_pipeline()
                    selected = "flux_dev"
                    fallback_device = getattr(pipe, "device", "cpu")
                    fallback_generator = make_generator(fallback_device if str(fallback_device) != "meta" else "cpu", request.seed)
                    result = pipe(
                        prompt=request.prompt,
                        num_inference_steps=request.num_inference_steps,
                        guidance_scale=request.guidance_scale,
                        width=request.width,
                        height=request.height,
                        generator=fallback_generator,
                    )
                    image = result.images[0]
                except Exception as fallback_exc:
                    logger.exception("Fallback to flux_dev also failed")
                    raise HTTPException(status_code=500, detail="Image generation failed") from fallback_exc
            else:
                logger.exception("Image generation failed for model %s", selected)
                raise HTTPException(status_code=500, detail="Image generation failed") from exc
        finally:
            release_gpu_coord(owner="backend")
            free_cuda_memory()
            # Brief pause to let CUDA allocator settle before next owner.
            time.sleep(0.5)

    return image, selected


def log_generation(
    db: Session,
    prompt: str,
    model_id: str,
    router_decision: RoutingDecision | None,
    image_path: str,
    thumbnail_path: str,
    prank_id: str | None = None,
    share_slug: str | None = None,
    session_id: str | None = None,
) -> str:
    log_entry = GenerationLog(
        prompt=prompt,
        model_id=model_id,
        router_json=json.dumps(router_decision.__dict__) if router_decision else None,
        image_path=image_path,
        thumbnail_path=thumbnail_path,
        prank_id=prank_id,
        share_slug=share_slug,
        session_id=session_id,
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)
    return str(log_entry.id)


def record_metric(
    db: Session,
    *,
    prompt: str,
    model_used: str,
    engine_requested: str | None,
    num_steps: int | None,
    guidance: float | None,
    width: int | None,
    height: int | None,
    seed: int | None,
    tf32_enabled: bool | None,
    is_synthetic: bool,
    is_prank: bool,
    queue_position: int | None,
    queue_wait_ms: int | None,
    duration_ms: int,
    started_at: datetime,
    ended_at: datetime,
    router_json: dict | None,
    session_id: str | None,
    share_slug: str | None,
    prompt_metadata: dict | None = None,
) -> None:
    """
    Persist a detailed metric row for latency/distribution analysis.
    """
    metric = GenerationMetric(
        prompt=prompt,
        prompt_length=len(prompt),
        model_used=model_used,
        engine_requested=engine_requested,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        width=width,
        height=height,
        seed=seed,
        tf32_enabled=tf32_enabled,
        is_synthetic=is_synthetic,
        is_prank=is_prank,
        queue_position_at_start=queue_position,
        queue_wait_ms=queue_wait_ms,
        duration_ms=duration_ms,
        started_at=started_at,
        ended_at=ended_at,
        router_json=json.dumps(router_json) if router_json else None,
        session_id=session_id,
        share_slug=share_slug,
        prompt_metadata=json.dumps(prompt_metadata) if prompt_metadata else None,
    )
    db.add(metric)
    db.commit()
    clear_cache()


def process_generation(
    request: TextGenerateRequest,
    db: Session,
    share_slug: str | None = None,
    prank_id: str | None = None,
) -> ImageResponse:
    # Clamp resolution for performance and VRAM headroom.
    if request.width and request.width > MAX_IMAGE_SIDE:
        request.width = MAX_IMAGE_SIDE
    if request.height and request.height > MAX_IMAGE_SIDE:
        request.height = MAX_IMAGE_SIDE

    start = time.time()
    if request.engine == "auto":
        decision = route_prompt(request.prompt)
        chosen_model = "flux_dev"
    else:
        chosen_model = "flux_dev"
        decision = RoutingDecision(
            chosen_model_id=chosen_model,
            scores={chosen_model: 1.0},
            tags=["manual"],
            reason="Forced flux_dev",
        )

    # Override steps/guidance with fast presets per model to ignore frontend sliders.
    preset = MODEL_PRESETS.get(chosen_model)
    if preset:
        request.num_inference_steps = int(preset["steps"])
        request.guidance_scale = float(preset["guidance"])

    image, actual_model = execute_model(chosen_model, request)
    image_path, thumb_path = save_generation_image(image)
    generation_time_ms = int((time.time() - start) * 1000)

    generation_id = log_generation(
        db=db,
        prompt=request.prompt,
        model_id=actual_model,
        router_decision=decision,
        image_path=image_path,
        thumbnail_path=thumb_path,
        prank_id=prank_id,
        share_slug=share_slug,
        session_id=request.session_id,
    )

    return ImageResponse(
        image_base64=pil_to_base64_png(image),
        generation_id=generation_id,
        model_id=actual_model,
        model_used=actual_model,
        thumbnail_base64=encode_thumbnail_base64(thumb_path),
        image_path=image_path,
        thumbnail_path=thumb_path,
        router_metadata=routing_metadata(decision),
        was_prank=False,
        matched_trigger_id=None,
        matched_trigger_text=None,
        generation_time_ms=generation_time_ms,
    )
