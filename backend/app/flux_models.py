"""Lazy-loaded FLUX text-to-image pipeline used by FastAPI."""

from __future__ import annotations

import logging
import threading
import os

import torch
from diffusers import FluxPipeline, StableDiffusion3Pipeline, StableDiffusionXLPipeline
try:
    from diffusers import HiDreamImagePipeline
except ImportError:
    # Fallback if HiDream not available in current diffusers version
    HiDreamImagePipeline = StableDiffusionXLPipeline
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM

from .config import (
    FLUX_TEXT_MODEL_ID,
    LOGO_SDXL_MODEL_ID,
    REALVIS_MODEL_ID,
    SD3_MODEL_ID,
    HIDREAM_TEXT_ENCODER_ID,
    get_device,
)

logger = logging.getLogger(__name__)

_text_pipeline: FluxPipeline | None = None
_realvis_pipeline: StableDiffusionXLPipeline | None = None
_sd3_pipeline: StableDiffusion3Pipeline | None = None
_logo_pipeline: HiDreamImagePipeline | StableDiffusionXLPipeline | None = None
_lock = threading.Lock()


def _disable_safety(pipe) -> None:
    """
    Disable safety checkers across pipelines to avoid NSFW filtering/black images.
    """
    try:
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
        if hasattr(pipe, "requires_safety_checker"):
            pipe.requires_safety_checker = False
        if hasattr(pipe, "feature_extractor"):
            pipe.feature_extractor = None
    except Exception:
        # Best-effort; pipeline APIs differ by model/version.
        pass


def unload_all_models_except(keep_model: str = None):
    """Unload all models except the specified one to free GPU memory."""
    global _text_pipeline, _realvis_pipeline, _sd3_pipeline, _logo_pipeline

    logger.info(f"Unloading all models except: {keep_model}")

    if keep_model != "flux" and _text_pipeline is not None:
        logger.info("Unloading FLUX pipeline")
        del _text_pipeline
        _text_pipeline = None

    if keep_model != "realvis" and _realvis_pipeline is not None:
        logger.info("Unloading RealVis pipeline")
        del _realvis_pipeline
        _realvis_pipeline = None

    if keep_model != "sd3" and _sd3_pipeline is not None:
        logger.info("Unloading SD3 pipeline")
        del _sd3_pipeline
        _sd3_pipeline = None

    if keep_model != "logo" and _logo_pipeline is not None:
        logger.info("Unloading Logo/HiDream pipeline")
        del _logo_pipeline
        _logo_pipeline = None

    # Aggressive memory clearing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(f"GPU memory cleared. Free memory: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")


def _load_text_pipeline() -> FluxPipeline:
    """Instantiate the text-to-image FLUX pipeline."""
    device = get_device()
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    logger.info("Loading FLUX text-to-image pipeline (%s) on %s", FLUX_TEXT_MODEL_ID, device)
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
        pipeline = FluxPipeline.from_pretrained(
            FLUX_TEXT_MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=token,
        )
        _disable_safety(pipeline)
        try:
            pipeline.to(device)
        except torch.cuda.OutOfMemoryError:
            logger.warning("Text pipeline OOM on GPU; enabling CPU offload.")
            pipeline.enable_model_cpu_offload()
        pipeline.enable_attention_slicing()
        pipeline.enable_vae_slicing()
    except Exception as exc:
        logger.exception("Failed to initialize FLUX text pipeline")
        raise RuntimeError(f"Failed to load FLUX text pipeline: {exc}") from exc
    return pipeline


def get_text_pipeline() -> FluxPipeline:
    """
    Return a singleton FLUX text-to-image pipeline.

    Pipelines are heavy to load, so we instantiate them once per process.
    """
    global _text_pipeline
    if _text_pipeline is None:
        with _lock:
            if _text_pipeline is None:
                _text_pipeline = _load_text_pipeline()
    return _text_pipeline


def _load_realvis_pipeline() -> StableDiffusionXLPipeline:
    device = get_device()
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    logger.info("Loading RealVisXL pipeline (%s) on %s", REALVIS_MODEL_ID, device)
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
        pipe = StableDiffusionXLPipeline.from_pretrained(
            REALVIS_MODEL_ID, torch_dtype=torch.bfloat16, token=token
        )
        _disable_safety(pipe)
        try:
            pipe.to(device)
        except torch.cuda.OutOfMemoryError:
            logger.warning("RealVis pipeline OOM on GPU; enabling CPU offload.")
            pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize RealVis pipeline")
        raise RuntimeError(f"Failed to load RealVis pipeline: {exc}") from exc
    return pipe


def get_realvis_pipeline() -> StableDiffusionXLPipeline:
    global _realvis_pipeline
    if _realvis_pipeline is None:
        with _lock:
            if _realvis_pipeline is None:
                _realvis_pipeline = _load_realvis_pipeline()
    return _realvis_pipeline


def _load_sd3_pipeline() -> StableDiffusion3Pipeline:
    device = get_device()
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    logger.info("Loading SD3-Medium pipeline (%s) on %s", SD3_MODEL_ID, device)
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
        pipe = StableDiffusion3Pipeline.from_pretrained(
            SD3_MODEL_ID, torch_dtype=torch.bfloat16, token=token
        )
        _disable_safety(pipe)
        try:
            pipe.to(device)
        except torch.cuda.OutOfMemoryError:
            logger.warning("SD3 pipeline OOM on GPU; enabling CPU offload.")
            pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize SD3 pipeline")
        raise RuntimeError(f"Failed to load SD3 pipeline: {exc}") from exc
    return pipe


def get_sd3_pipeline() -> StableDiffusion3Pipeline:
    global _sd3_pipeline
    if _sd3_pipeline is None:
        with _lock:
            if _sd3_pipeline is None:
                _sd3_pipeline = _load_sd3_pipeline()
    return _sd3_pipeline


def _load_logo_pipeline() -> HiDreamImagePipeline | StableDiffusionXLPipeline:
    device = get_device()
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    logger.info("Loading logo/text pipeline (%s) on %s", LOGO_SDXL_MODEL_ID, device)

    try:
        # Check if we're using HiDream model
        if "HiDream" in LOGO_SDXL_MODEL_ID:
            logger.info("Initializing HiDream pipeline with Llama text encoder")

            # CRITICAL: Unload all other models to make room for HiDream
            logger.info("Unloading other models to make room for HiDream (large params)")
            unload_all_models_except("logo")

            # Clear CUDA cache before loading
            if device == "cuda":
                logger.info("Clearing CUDA cache before HiDream load...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                free_mem = torch.cuda.mem_get_info()[0] / 1024**3
                logger.info(f"Available GPU memory before load: {free_mem:.2f} GB")

            # Load Llama text encoder for HiDream with memory optimization
            logger.info("Loading Llama-3.1-8B text encoder...")
            tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
                HIDREAM_TEXT_ENCODER_ID,
                token=token
            )

            # Try to load with lower precision or quantization
            try:
                text_encoder_4 = LlamaForCausalLM.from_pretrained(
                    HIDREAM_TEXT_ENCODER_ID,
                    output_hidden_states=True,
                    output_attentions=True,
                    torch_dtype=torch.float16,  # Use fp16 instead of bfloat16
                    token=token,
                    low_cpu_mem_usage=True,
                    device_map="balanced"  # Use balanced strategy for multi-GPU or CPU offload
                )
            except Exception as e:
                logger.warning(f"Failed to load text encoder with fp16: {e}")
                # Fallback to CPU loading
                text_encoder_4 = LlamaForCausalLM.from_pretrained(
                    HIDREAM_TEXT_ENCODER_ID,
                    output_hidden_states=True,
                    output_attentions=True,
                    torch_dtype=torch.bfloat16,
                    token=token,
                    low_cpu_mem_usage=True
                )

            # Clear cache again after text encoder
            if device == "cuda":
                torch.cuda.empty_cache()

            # Load HiDream pipeline with optimizations
            logger.info("Loading HiDream-I1-Full pipeline...")
            pipe = HiDreamImagePipeline.from_pretrained(
                LOGO_SDXL_MODEL_ID,
                tokenizer_4=tokenizer_4,
                text_encoder_4=text_encoder_4,
                torch_dtype=torch.float16,  # Use fp16 for smaller size
                token=token,
                low_cpu_mem_usage=True
                # Don't use device_map here - we'll handle device placement manually
            )

            # Disable safety checker for HiDream; logo/text prompts get blacked out otherwise.
            _disable_safety(pipe)

            # Force CPU offload for HiDream - it's too large for GPU
            if device == "cuda":
                logger.info("HiDream is 17B params - using sequential CPU offload strategy")
                logger.info("Model weights will be in RAM, GPU used for computation only")

                # Use sequential CPU offload - this moves layers to GPU one at a time
                try:
                    pipe.enable_sequential_cpu_offload()
                    logger.info("✓ HiDream loaded with sequential CPU offload")
                except AttributeError:
                    # Fallback if sequential not available
                    logger.warning("Sequential offload not available, using standard CPU offload")
                    pipe.enable_model_cpu_offload()
                    logger.info("✓ HiDream loaded with model CPU offload")
            else:
                pipe.to(device)

        else:
            # Fallback to SDXL pipeline for backward compatibility
            logger.info("Using SDXL pipeline for logo/text generation")
            pipe = StableDiffusionXLPipeline.from_pretrained(
                LOGO_SDXL_MODEL_ID, torch_dtype=torch.bfloat16, token=token
            )

            _disable_safety(pipe)
            try:
                pipe.to(device)
            except torch.cuda.OutOfMemoryError:
                logger.warning("Logo pipeline OOM on GPU; enabling CPU offload.")
                pipe.enable_model_cpu_offload()

        # Memory optimizations
        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing()
        if hasattr(pipe, 'enable_vae_slicing'):
            pipe.enable_vae_slicing()

        # Final memory clear
        if device == "cuda":
            torch.cuda.empty_cache()
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            logger.info(f"GPU memory after HiDream load: {free_mem:.2f} GB free")

    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize logo/text pipeline")
        raise RuntimeError(f"Failed to load logo/text pipeline: {exc}") from exc

    return pipe


def get_logo_pipeline() -> HiDreamImagePipeline | StableDiffusionXLPipeline:
    global _logo_pipeline
    if _logo_pipeline is None:
        with _lock:
            if _logo_pipeline is None:
                # For HiDream, always unload other models first due to size
                if "HiDream" in LOGO_SDXL_MODEL_ID:
                    logger.info("Preparing to load HiDream - unloading other models first")
                    unload_all_models_except("logo")
                _logo_pipeline = _load_logo_pipeline()
    return _logo_pipeline
