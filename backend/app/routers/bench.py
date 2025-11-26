"""Nerd bench (5090 test) endpoints and background worker."""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
import uuid
from pathlib import Path

import torch
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..database import SessionLocal, get_db
from ..flux_models import (
    get_flux2_pipeline,
    get_realvis_pipeline,
    get_sd3_pipeline,
    get_text_pipeline,
    unload_all_models_except,
    _remote_text_encoder,
)
from ..models import BenchRun, BenchRunResult
from ..queue_manager import QueueFull
from ..schemas import (
    BenchEngineSettings,
    NerdBenchEnqueueResponse,
    NerdBenchEngineStatus,
    NerdBenchRequest,
    NerdBenchStatus,
)
from ..services.deps import (
    BENCH_WARMUP_PROMPT,
    GEN_QUEUE,
    MAX_IMAGE_SIDE,
    file_media_type,
    free_cuda_memory,
    make_generator,
    restore_tf32,
    set_tf32,
)
from ..storage import save_bench_image

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/bench", tags=["bench"])

BENCH_PROMPTS: list[str] = [
    "a cinematic portrait of a woman in neon cyberpunk lighting, detailed skin texture",
    "a lush landscape of mountains and a lake at sunrise, ultra wide angle",
    "a futuristic UI dashboard mockup with clean typography and iconography",
    "a photorealistic cat wearing sunglasses on a city street",
    "a watercolor painting of a cozy coffee shop interior with warm light",
]


# ---------------------------------------------------------------------------
# GPU Metrics Helpers
# ---------------------------------------------------------------------------


def _get_gpu_info() -> dict:
    """Get GPU hardware info for data science."""
    info = {
        "gpu_name": None,
        "gpu_compute_capability": None,
        "gpu_total_mem_mb": None,
    }
    if torch.cuda.is_available():
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            info["gpu_compute_capability"] = f"{cap[0]}.{cap[1]}"
            info["gpu_total_mem_mb"] = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
        except Exception:
            pass
    return info


def _get_gpu_memory_mb() -> dict:
    """Get current GPU memory usage in MB."""
    mem = {
        "allocated_mb": None,
        "reserved_mb": None,
        "free_mb": None,
    }
    if torch.cuda.is_available():
        try:
            mem["allocated_mb"] = int(torch.cuda.memory_allocated() / (1024 * 1024))
            mem["reserved_mb"] = int(torch.cuda.memory_reserved() / (1024 * 1024))
            free, total = torch.cuda.mem_get_info()
            mem["free_mb"] = int(free / (1024 * 1024))
        except Exception:
            pass
    return mem


def _get_software_versions() -> dict:
    """Get software versions for reproducibility."""
    import sys
    versions = {
        "torch_version": None,
        "diffusers_version": None,
        "python_version": None,
    }
    try:
        versions["torch_version"] = torch.__version__
    except Exception:
        pass
    try:
        import diffusers
        versions["diffusers_version"] = diffusers.__version__
    except Exception:
        pass
    try:
        versions["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    except Exception:
        pass
    return versions


def _get_model_metadata(engine: str) -> dict:
    """Get model-specific metadata."""
    metadata = {
        "model_dtype": "bfloat16",
        "quantization": "none",
        "uses_remote_encoder": False,
    }
    if engine == "flux2_dev":
        metadata["quantization"] = "4bit"
        metadata["uses_remote_encoder"] = True
    elif engine == "hidream_dev":
        metadata["model_dtype"] = "float16"
    return metadata


def _bench_record_run(
    db: Session,
    run_id: str,
    engine: str,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    seed: int,
    elapsed_ms: int,
    tf32: bool,
    image_path: str,
    # Extended metrics
    model_load_ms: int | None = None,
    warmup_ms: int | None = None,
    text_encoder_ms: int | None = None,
    inference_only_ms: int | None = None,
    gpu_mem_before_load_mb: int | None = None,
    gpu_mem_after_load_mb: int | None = None,
    gpu_mem_peak_mb: int | None = None,
    gpu_mem_after_inference_mb: int | None = None,
    gpu_mem_allocated_mb: int | None = None,
    gpu_mem_reserved_mb: int | None = None,
    gpu_name: str | None = None,
    gpu_compute_capability: str | None = None,
    gpu_total_mem_mb: int | None = None,
    torch_version: str | None = None,
    diffusers_version: str | None = None,
    python_version: str | None = None,
    model_dtype: str | None = None,
    quantization: str | None = None,
    uses_remote_encoder: bool | None = None,
) -> None:
    rec = BenchRunResult(
        run_id=run_id,
        engine=engine,
        steps=steps,
        guidance=guidance,
        width=width,
        height=height,
        seed=seed,
        elapsed_ms=elapsed_ms,
        tf32_enabled=tf32,
        image_path=image_path,
        # Extended metrics
        model_load_ms=model_load_ms,
        warmup_ms=warmup_ms,
        text_encoder_ms=text_encoder_ms,
        inference_only_ms=inference_only_ms,
        gpu_mem_before_load_mb=gpu_mem_before_load_mb,
        gpu_mem_after_load_mb=gpu_mem_after_load_mb,
        gpu_mem_peak_mb=gpu_mem_peak_mb,
        gpu_mem_after_inference_mb=gpu_mem_after_inference_mb,
        gpu_mem_allocated_mb=gpu_mem_allocated_mb,
        gpu_mem_reserved_mb=gpu_mem_reserved_mb,
        gpu_name=gpu_name,
        gpu_compute_capability=gpu_compute_capability,
        gpu_total_mem_mb=gpu_total_mem_mb,
        torch_version=torch_version,
        diffusers_version=diffusers_version,
        python_version=python_version,
        model_dtype=model_dtype,
        quantization=quantization,
        uses_remote_encoder=uses_remote_encoder,
    )
    db.add(rec)
    db.commit()


# ---------------------------------------------------------------------------
# Background Worker
# ---------------------------------------------------------------------------


def _run_bench_worker(run_id: str, engines_payload: list[BenchEngineSettings], resolution: int | None, tf32_default: bool | None):
    """Background worker to process a bench run using the shared queue."""
    db = SessionLocal()
    job = None
    try:
        run = db.query(BenchRun).filter(BenchRun.id == run_id).first()
        if not run:
            return
        session_id = run.session_id or "anon"
        try:
            job = GEN_QUEUE.enqueue(session_id, job_type="bench", generation_id=run_id)
        except QueueFull:
            run.status = "error"
            db.commit()
            return
        GEN_QUEUE.wait_for_turn(job["generation_id"])

        # Capture GPU info and software versions once at start
        gpu_info = _get_gpu_info()
        sw_versions = _get_software_versions()

        free_cuda_memory()
        try:
            unload_all_models_except(None)
        except Exception:
            pass
        warm_prompt = BENCH_WARMUP_PROMPT

        # Reorder engines to run flux2_dev FIRST - it's a 4-bit quantized model
        # that needs fresh GPU memory (~17GB). Other models should run after,
        # even if FLUX.2 doesn't fully release its memory (they can use CPU offload).
        flux2_first = [e for e in engines_payload if e.engine == "flux2_dev"]
        others = [e for e in engines_payload if e.engine != "flux2_dev"]
        engines_payload = flux2_first + others
        logger.info(f"Engine order: {[e.engine for e in engines_payload]}")

        for eng_cfg in engines_payload:
            engine = eng_cfg.engine
            steps = int(eng_cfg.steps)
            guidance = float(eng_cfg.guidance)
            width = eng_cfg.width or resolution or 512
            height = eng_cfg.height or resolution or 512
            width = min(width, MAX_IMAGE_SIDE)
            height = min(height, MAX_IMAGE_SIDE)
            tf32_flag = eng_cfg.tf32 if eng_cfg.tf32 is not None else (tf32_default if tf32_default is not None else True)

            # Clear memory before each model for accurate metrics
            try:
                unload_all_models_except(None)
                free_cuda_memory()
            except Exception:
                pass

            # Initialize metrics for this engine
            metrics = {
                "model_load_ms": None,
                "warmup_ms": None,
                "text_encoder_ms": None,
                "inference_only_ms": None,
                "gpu_mem_before_load_mb": None,
                "gpu_mem_after_load_mb": None,
                "gpu_mem_peak_mb": None,
                "gpu_mem_after_inference_mb": None,
                "gpu_mem_allocated_mb": None,
                "gpu_mem_reserved_mb": None,
            }
            model_metadata = _get_model_metadata(engine)

            # Capture memory before load
            mem_before = _get_gpu_memory_mb()
            metrics["gpu_mem_before_load_mb"] = mem_before.get("allocated_mb")

            # Loading phase
            run.status = "loading_model"
            run.current_engine = engine
            db.commit()

            prev_tf32 = set_tf32(tf32_flag)

            # Time model loading
            load_start = time.perf_counter()
            try:
                if engine == "flux_dev":
                    pipe = get_text_pipeline()
                elif engine == "realvis_xl":
                    pipe = get_realvis_pipeline()
                elif engine == "sd3_medium":
                    pipe = get_sd3_pipeline()
                elif engine == "flux2_dev":
                    pipe = get_flux2_pipeline()
                else:
                    restore_tf32(*prev_tf32)
                    continue
            except Exception as load_exc:
                logger.exception("Model load failed for engine %s: %s", engine, load_exc)
                restore_tf32(*prev_tf32)
                # Record failed engine result and continue with other engines
                result = BenchResult(
                    run_id=run_id,
                    engine=engine,
                    status="error",
                    error_message=f"Model load failed: {load_exc}",
                    steps=steps,
                    guidance=guidance,
                    width=width,
                    height=height,
                    tf32_enabled=tf32_flag,
                )
                db.add(result)
                db.commit()
                continue
            load_end = time.perf_counter()
            metrics["model_load_ms"] = int((load_end - load_start) * 1000)

            # Capture memory after load
            mem_after_load = _get_gpu_memory_mb()
            metrics["gpu_mem_after_load_mb"] = mem_after_load.get("allocated_mb")

            # Reset peak memory tracking
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # Warmup phase (timed for metrics)
            warmup_start = time.perf_counter()
            try:
                if engine == "flux2_dev":
                    # FLUX2 requires remote text encoder for prompt embeddings
                    prompt_embeds = _remote_text_encoder(warm_prompt, str(pipe.device))
                    _ = pipe(
                        prompt_embeds=prompt_embeds,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        width=width,
                        height=height,
                        generator=make_generator(pipe.device, None),
                    )
                else:
                    _ = pipe(
                        warm_prompt,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        width=width,
                        height=height,
                        generator=make_generator(pipe.device, None),
                    )
            except Exception as warmup_exc:
                logger.exception("Warmup failed for engine %s: %s", engine, warmup_exc)
                restore_tf32(*prev_tf32)
                # Record failed engine result and continue with other engines
                result = BenchResult(
                    run_id=run_id,
                    engine=engine,
                    status="error",
                    error_message=f"Warmup failed: {warmup_exc}",
                    steps=steps,
                    guidance=guidance,
                    width=width,
                    height=height,
                    tf32_enabled=tf32_flag,
                )
                db.add(result)
                db.commit()
                continue
            warmup_end = time.perf_counter()
            metrics["warmup_ms"] = int((warmup_end - warmup_start) * 1000)

            # Reset peak memory for actual inference
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # Timing phase
            run.status = "running"
            run.current_engine = engine
            db.commit()

            seed = secrets.randbelow(2**31 - 1)
            gen = make_generator(pipe.device, seed)

            # Time text encoder and inference separately for FLUX2
            text_encoder_ms = 0
            inference_start = time.perf_counter()
            try:
                if engine == "flux2_dev":
                    # Time remote text encoder separately
                    te_start = time.perf_counter()
                    prompt_embeds = _remote_text_encoder(run.prompt, str(pipe.device))
                    te_end = time.perf_counter()
                    text_encoder_ms = int((te_end - te_start) * 1000)
                    metrics["text_encoder_ms"] = text_encoder_ms

                    # Time pure inference
                    inf_start = time.perf_counter()
                    out = pipe(
                        prompt_embeds=prompt_embeds,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        width=width,
                        height=height,
                        generator=gen,
                    )
                    inf_end = time.perf_counter()
                    metrics["inference_only_ms"] = int((inf_end - inf_start) * 1000)
                else:
                    inf_start = time.perf_counter()
                    out = pipe(
                        run.prompt,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        width=width,
                        height=height,
                        generator=gen,
                    )
                    inf_end = time.perf_counter()
                    metrics["inference_only_ms"] = int((inf_end - inf_start) * 1000)
                image = out.images[0]
            except Exception as infer_exc:
                logger.exception("Inference failed for engine %s: %s", engine, infer_exc)
                restore_tf32(*prev_tf32)
                run.status = "error"
                db.commit()
                break

            # Calculate total elapsed time from inference start
            elapsed_ms = int((time.perf_counter() - inference_start) * 1000)

            # Capture peak memory after inference
            if torch.cuda.is_available():
                try:
                    metrics["gpu_mem_peak_mb"] = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
                except Exception:
                    pass

            # Capture final memory state
            mem_after = _get_gpu_memory_mb()
            metrics["gpu_mem_after_inference_mb"] = mem_after.get("allocated_mb")
            metrics["gpu_mem_allocated_mb"] = mem_after.get("allocated_mb")
            metrics["gpu_mem_reserved_mb"] = mem_after.get("reserved_mb")

            restore_tf32(*prev_tf32)

            image_path = save_bench_image(str(run.id), engine, image)
            _bench_record_run(
                db,
                run_id=str(run.id),
                engine=engine,
                steps=steps,
                guidance=guidance,
                width=width,
                height=height,
                seed=seed,
                elapsed_ms=elapsed_ms,
                tf32=tf32_flag,
                image_path=image_path,
                # Extended metrics
                model_load_ms=metrics["model_load_ms"],
                warmup_ms=metrics["warmup_ms"],
                text_encoder_ms=metrics["text_encoder_ms"],
                inference_only_ms=metrics["inference_only_ms"],
                gpu_mem_before_load_mb=metrics["gpu_mem_before_load_mb"],
                gpu_mem_after_load_mb=metrics["gpu_mem_after_load_mb"],
                gpu_mem_peak_mb=metrics["gpu_mem_peak_mb"],
                gpu_mem_after_inference_mb=metrics["gpu_mem_after_inference_mb"],
                gpu_mem_allocated_mb=metrics["gpu_mem_allocated_mb"],
                gpu_mem_reserved_mb=metrics["gpu_mem_reserved_mb"],
                # GPU info
                gpu_name=gpu_info.get("gpu_name"),
                gpu_compute_capability=gpu_info.get("gpu_compute_capability"),
                gpu_total_mem_mb=gpu_info.get("gpu_total_mem_mb"),
                # Software versions
                torch_version=sw_versions.get("torch_version"),
                diffusers_version=sw_versions.get("diffusers_version"),
                python_version=sw_versions.get("python_version"),
                # Model metadata
                model_dtype=model_metadata.get("model_dtype"),
                quantization=model_metadata.get("quantization"),
                uses_remote_encoder=model_metadata.get("uses_remote_encoder"),
            )

        run.status = "done" if run.status != "error" else run.status
        run.current_engine = None
        db.commit()

        # Warm flux back up for live traffic
        try:
            unload_all_models_except("flux")
            free_cuda_memory()
            get_text_pipeline()
        except Exception:
            logger.warning("Failed to warm flux after bench")
    finally:
        if job:
            GEN_QUEUE.release(job["generation_id"], success=(run.status == "done"), payload={"bench_run_id": str(run_id)})
        db.close()


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@router.post("/5090", response_model=NerdBenchEnqueueResponse)
def enqueue_nerd_bench(
    payload: NerdBenchRequest,
    db: Session = Depends(get_db),
) -> NerdBenchEnqueueResponse:
    """
    Enqueue a nerd bench run. Returns run_id immediately; poll GET /api/bench/5090/{run_id}.
    """
    # Basic validation
    if not payload.prompt or not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty")
    session_id = payload.session_id or "anon"
    prompt = payload.prompt.strip()
    prompt_length = len(prompt)
    resolution = payload.resolution or 512
    if resolution not in (512, 1024):
        raise HTTPException(status_code=400, detail="Resolution must be 512 or 1024")
    resolution = min(resolution, MAX_IMAGE_SIDE)
    tf32_default = True if payload.tf32_enabled is None else bool(payload.tf32_enabled)

    # Per-engine validation
    PER_MODEL_LIMITS = {
        "flux_dev": {"steps_min": 1, "steps_max": 5, "guidance_min": 0.0, "guidance_max": 2.0},
        "realvis_xl": {"steps_min": 8, "steps_max": 48, "guidance_min": 1.0, "guidance_max": 8.0},
        "sd3_medium": {"steps_min": 18, "steps_max": 48, "guidance_min": 3.0, "guidance_max": 9.0},
        "flux2_dev": {"steps_min": 20, "steps_max": 28, "guidance_min": 3.5, "guidance_max": 4.5},
    }
    for eng in payload.engines:
        limits = PER_MODEL_LIMITS.get(eng.engine)
        if not limits:
            raise HTTPException(status_code=400, detail=f"Unsupported engine {eng.engine}")
        if not (limits["steps_min"] <= eng.steps <= limits["steps_max"]):
            raise HTTPException(
                status_code=400,
                detail=f"{eng.engine} steps must be between {limits['steps_min']} and {limits['steps_max']}",
            )
        if not (limits["guidance_min"] <= eng.guidance <= limits["guidance_max"]):
            raise HTTPException(
                status_code=400,
                detail=f"{eng.engine} guidance must be between {limits['guidance_min']} and {limits['guidance_max']}",
            )

    bench_run = BenchRun(
        session_id=session_id,
        prompt=prompt,
        prompt_length=prompt_length,
        status="queued",
        current_engine=None,
        resolution=resolution,
        tf32_enabled=tf32_default,
        engines_json=json.dumps([e.model_dump() for e in payload.engines]),
    )
    db.add(bench_run)
    db.commit()
    db.refresh(bench_run)

    # Start background worker
    try:
        threading.Thread(
            target=_run_bench_worker,
            args=(str(bench_run.id), payload.engines, resolution, tf32_default),
            daemon=True,
        ).start()
    except Exception as exc:
        logger.exception("Failed to start bench worker: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to start benchmark") from exc

    return NerdBenchEnqueueResponse(
        run_id=str(bench_run.id),
        prompt=prompt,
        prompt_length=prompt_length,
    )


@router.get("/5090/{run_id}", response_model=NerdBenchStatus)
def get_nerd_bench_status(run_id: str, db: Session = Depends(get_db)) -> NerdBenchStatus:
    try:
        run_uuid = uuid.UUID(str(run_id))
    except Exception:
        raise HTTPException(status_code=404, detail="Run not found")

    try:
        run = db.query(BenchRun).filter(BenchRun.id == run_uuid).first()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        engines_cfg = []
        try:
            if run.engines_json:
                engines_cfg = json.loads(run.engines_json)
        except Exception:
            engines_cfg = []

        results = db.query(BenchRunResult).filter(BenchRunResult.run_id == run.id).all()
        by_engine: dict[str, NerdBenchEngineStatus] = {}

        # Build a lookup for desired params from cfg
        desired_params = {e.get("engine"): e for e in engines_cfg if isinstance(e, dict)}

        for engine in ["flux_dev", "realvis_xl", "sd3_medium", "flux2_dev"]:
            res = next((r for r in results if r.engine == engine), None)
            if res:
                by_engine[engine] = NerdBenchEngineStatus(
                    status="done",
                    elapsed_ms=res.elapsed_ms,
                    image_url=f"/api/bench/images/{res.id}",
                    steps=res.steps,
                    guidance=res.guidance,
                    width=res.width,
                    height=res.height,
                    seed=res.seed,
                    tf32=res.tf32_enabled,
                    tf32_enabled=res.tf32_enabled,
                )
            else:
                desired = desired_params.get(engine, {})
                if run.current_engine == engine:
                    if run.status == "loading_model":
                        status = "loading_model"
                    elif run.status == "running":
                        status = "running"
                    else:
                        status = "pending"
                else:
                    status = "pending"
                by_engine[engine] = NerdBenchEngineStatus(
                    status=status,
                    steps=desired.get("steps"),
                    guidance=desired.get("guidance"),
                    width=desired.get("width") or run.resolution,
                    height=desired.get("height") or run.resolution,
                    tf32=desired.get("tf32") if desired.get("tf32") is not None else run.tf32_enabled,
                    tf32_enabled=desired.get("tf32") if desired.get("tf32") is not None else run.tf32_enabled,
                )

        return NerdBenchStatus(
            run_id=str(run.id),
            prompt=run.prompt,
            prompt_length=run.prompt_length,
            status=run.status,
            current_engine=run.current_engine,
            resolution=run.resolution,
            tf32_enabled=run.tf32_enabled,
            engines=by_engine,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Bench status failure for run_id=%s: %s", run_id, exc)
        raise HTTPException(status_code=500, detail="Bench status unavailable") from exc


@router.get("/images/{result_id}")
def get_bench_image(result_id: str, db: Session = Depends(get_db)):
    res = db.query(BenchRunResult).filter(BenchRunResult.id == result_id).first()
    if not res:
        raise HTTPException(status_code=404, detail="Bench image not found")
    path = Path(res.image_path)
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail="Bench image file missing")
    return FileResponse(path, media_type=file_media_type(path))
