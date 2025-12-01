"""Lazy-loaded FLUX text-to-image pipeline used by FastAPI."""

from __future__ import annotations

import io
import logging
import threading
import os

import requests
import torch
from diffusers import FluxPipeline, Flux2Pipeline, StableDiffusion3Pipeline, StableDiffusionXLPipeline
try:
    from diffusers import HiDreamImagePipeline
except ImportError:
    # Fallback if HiDream not available in current diffusers version
    HiDreamImagePipeline = StableDiffusionXLPipeline
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM

from .config import (
    FLUX_TEXT_MODEL_ID,
    FLUX2_4BIT_MODEL_ID,
    FLUX2_REMOTE_TEXT_ENCODER_URL,
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
_flux2_pipeline: FluxPipeline | None = None
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


def unload_current_model():
    """Unload whichever model is currently loaded to free GPU and RAM."""
    global _text_pipeline, _realvis_pipeline, _sd3_pipeline, _logo_pipeline, _flux2_pipeline
    import gc

    logger.info("Unloading current model to free GPU/RAM...")

    # Find and unload whichever pipeline is loaded
    if _text_pipeline is not None:
        logger.info("Unloading FLUX pipeline")
        del _text_pipeline
        _text_pipeline = None
    if _realvis_pipeline is not None:
        logger.info("Unloading RealVis pipeline")
        del _realvis_pipeline
        _realvis_pipeline = None
    if _sd3_pipeline is not None:
        logger.info("Unloading SD3 pipeline")
        del _sd3_pipeline
        _sd3_pipeline = None
    if _logo_pipeline is not None:
        logger.info("Unloading Logo/HiDream pipeline")
        del _logo_pipeline
        _logo_pipeline = None
    if _flux2_pipeline is not None:
        logger.info("Unloading FLUX.2 pipeline")
        del _flux2_pipeline
        _flux2_pipeline = None

    # Aggressive cleanup
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        logger.info(f"After unload: GPU has {free_mem:.2f} GB free")


def unload_all_models_except(keep_model: str | None = None):
    """Unload all models except the specified one to free GPU memory."""
    global _text_pipeline, _realvis_pipeline, _sd3_pipeline, _logo_pipeline, _flux2_pipeline

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

    if keep_model != "flux2" and _flux2_pipeline is not None:
        logger.info("Unloading FLUX.2 pipeline - using aggressive cleanup for bitsandbytes 4-bit")
        # bitsandbytes 4-bit models need to be moved to CPU first to release GPU memory
        try:
            # Move entire pipeline to CPU first - this should release GPU memory for 4-bit weights
            logger.info("Moving FLUX.2 to CPU before deletion...")
            _flux2_pipeline.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                free_mem = torch.cuda.mem_get_info()[0] / 1024**3
                logger.info(f"GPU memory after FLUX.2 to CPU: {free_mem:.2f} GB")
        except Exception as exc:
            logger.warning("Failed to move FLUX.2 to CPU: %s", exc)
        # Now delete the pipeline
        del _flux2_pipeline
        _flux2_pipeline = None
        # Extra gc passes for bitsandbytes cleanup
        import gc
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # Aggressive memory clearing
    import gc
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        except Exception:
            pass
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.info(f"GPU memory cleared. Free memory: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")


def _load_text_pipeline() -> FluxPipeline:
    """Instantiate the text-to-image FLUX pipeline directly on GPU."""
    device = get_device()
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    logger.info("Loading FLUX text-to-image pipeline (%s) DIRECTLY to %s", FLUX_TEXT_MODEL_ID, device)
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            logger.info(f"GPU memory before FLUX load: {free_mem:.2f} GB free")

        # Load directly to GPU - no CPU staging
        pipeline = FluxPipeline.from_pretrained(
            FLUX_TEXT_MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=token,
            device_map="balanced" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        _disable_safety(pipeline)

        # Only move if not using device_map
        if device == "cuda" and not hasattr(pipeline, 'hf_device_map'):
            pipeline.to(device)

        pipeline.enable_attention_slicing()
        pipeline.enable_vae_slicing()

        if device == "cuda":
            torch.cuda.synchronize()
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            logger.info(f"GPU memory after FLUX load: {free_mem:.2f} GB free")
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
    logger.info("Loading RealVisXL pipeline (%s) DIRECTLY to %s", REALVIS_MODEL_ID, device)
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            logger.info(f"GPU memory before RealVis load: {free_mem:.2f} GB free")

        pipe = StableDiffusionXLPipeline.from_pretrained(
            REALVIS_MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=token,
            device_map="balanced" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        _disable_safety(pipe)

        if device == "cuda" and not hasattr(pipe, 'hf_device_map'):
            pipe.to(device)

        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

        if device == "cuda":
            torch.cuda.synchronize()
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            logger.info(f"GPU memory after RealVis load: {free_mem:.2f} GB free")
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
    logger.info("Loading SD3-Medium pipeline (%s) DIRECTLY to %s", SD3_MODEL_ID, device)
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            logger.info(f"GPU memory before SD3 load: {free_mem:.2f} GB free")

        pipe = StableDiffusion3Pipeline.from_pretrained(
            SD3_MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=token,
            device_map="balanced" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        _disable_safety(pipe)

        if device == "cuda" and not hasattr(pipe, 'hf_device_map'):
            pipe.to(device)

        pipe.enable_attention_slicing()

        if device == "cuda":
            torch.cuda.synchronize()
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            logger.info(f"GPU memory after SD3 load: {free_mem:.2f} GB free")
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


def _remote_text_encoder(prompt: str, device: str = "cuda") -> torch.Tensor:
    """
    Call the official remote text encoder for FLUX.2 [dev].
    Returns prompt_embeds tensor for use with Flux2Pipeline.
    """
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_HUB_TOKEN or HF_TOKEN required for FLUX.2 remote text encoder")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    logger.info("Calling remote text encoder for FLUX.2...")
    response = requests.post(
        FLUX2_REMOTE_TEXT_ENCODER_URL,
        json={"prompt": [prompt]},
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()

    # The endpoint returns serialized torch tensors
    data = torch.load(io.BytesIO(response.content), map_location=device)
    logger.info("Remote text encoder returned embeddings successfully")

    # Handle different return formats from the endpoint
    logger.info("Remote text encoder data type: %s", type(data))
    if isinstance(data, dict):
        logger.info("Remote text encoder data keys: %s", list(data.keys()))
        prompt_embeds = data.get("prompt_embeds", data.get("embeddings"))
    elif isinstance(data, (list, tuple)) and len(data) >= 1:
        prompt_embeds = data[0]
    elif isinstance(data, torch.Tensor):
        # FLUX.2 remote text encoder returns a single Tensor that is the prompt_embeds
        logger.info("Remote text encoder returned Tensor shape: %s, dtype: %s", data.shape, data.dtype)
        prompt_embeds = data
    else:
        logger.warning("Unexpected data format from remote text encoder: %s", type(data))
        raise RuntimeError(f"Unexpected remote text encoder format: {type(data)}")

    if prompt_embeds is None:
        raise RuntimeError(
            f"Remote text encoder did not return prompt_embeds. "
            f"Data type: {type(data)}, keys: {data.keys() if isinstance(data, dict) else 'N/A'}"
        )

    logger.info("prompt_embeds shape: %s", prompt_embeds.shape if hasattr(prompt_embeds, 'shape') else 'N/A')

    return prompt_embeds


def _load_flux2_pipeline() -> Flux2Pipeline:
    """
    Load the FLUX.2-dev 4-bit quantized pipeline.
    Uses remote text encoder for prompt embeddings.
    """
    device = get_device()
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    logger.info("Loading FLUX.2-dev 4-bit pipeline (%s) on %s", FLUX2_4BIT_MODEL_ID, device)

    try:
        # Unload other models first - FLUX.2 is large
        logger.info("Unloading other models to make room for FLUX.2")
        unload_all_models_except("flux2")

        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            logger.info(f"Available GPU memory before FLUX.2 load: {free_mem:.2f} GB")

        # Load the 4-bit quantized pipeline using Flux2Pipeline
        # We use remote text encoder API instead to save ~10GB VRAM
        # Note: bitsandbytes 4-bit doesn't support device_map with CPU offload
        pipeline = Flux2Pipeline.from_pretrained(
            FLUX2_4BIT_MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=token,
            text_encoder=None,
            tokenizer=None,
        )

        _disable_safety(pipeline)

        # Move to GPU - 4-bit model should fit in ~15GB
        try:
            pipeline.to(device)
            logger.info("FLUX.2 loaded directly to GPU")
        except Exception as exc:
            logger.warning("FLUX.2 direct GPU load failed (%s); trying sequential CPU offload", exc)
            try:
                pipeline.enable_sequential_cpu_offload()
                logger.info("FLUX.2 loaded with sequential CPU offload")
            except Exception as e2:
                logger.warning("Sequential CPU offload also failed: %s", e2)
                pipeline.enable_model_cpu_offload()
                logger.info("FLUX.2 loaded with model CPU offload")

        if hasattr(pipeline, 'enable_attention_slicing'):
            pipeline.enable_attention_slicing()
        if hasattr(pipeline, 'enable_vae_slicing'):
            pipeline.enable_vae_slicing()

        if device == "cuda":
            torch.cuda.empty_cache()
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            logger.info(f"GPU memory after FLUX.2 load: {free_mem:.2f} GB free")

    except Exception as exc:
        logger.exception("Failed to initialize FLUX.2 pipeline")
        raise RuntimeError(f"Failed to load FLUX.2 pipeline: {exc}") from exc

    return pipeline


def get_flux2_pipeline() -> FluxPipeline:
    """
    Return a singleton FLUX.2-dev 4-bit pipeline.
    """
    global _flux2_pipeline
    if _flux2_pipeline is None:
        with _lock:
            if _flux2_pipeline is None:
                logger.info("Preparing to load FLUX.2 - unloading other models first")
                unload_all_models_except("flux2")
                _flux2_pipeline = _load_flux2_pipeline()
    return _flux2_pipeline


def generate_flux2(
    prompt: str,
    num_inference_steps: int = 24,
    guidance_scale: float = 4.0,
    width: int = 768,
    height: int = 768,
    seed: int | None = None,
) -> "PIL.Image.Image":
    """
    Generate an image using FLUX.2-dev with remote text encoder.
    This is a convenience wrapper that handles the remote text encoder call.
    """
    import PIL.Image

    pipe = get_flux2_pipeline()
    device = pipe.device if hasattr(pipe, 'device') else get_device()

    # Get embeddings from remote text encoder
    prompt_embeds, pooled_prompt_embeds = _remote_text_encoder(prompt, str(device))

    # Create generator for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    # Generate with prompt embeddings
    result = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )

    return result.images[0]
