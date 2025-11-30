#!/usr/bin/env python3
"""
Rapid data collection script for RealVisXL parameter sweep.
Generates 600 images with randomized steps/guidance for ML analysis.

Run with Docker backend STOPPED (to avoid GPU conflicts):
  docker compose stop backend
  cd /home/halsted/Python/flexy-face
  DATABASE_URL="postgresql://flexyface:flexypass@localhost:7432/flexyface" \
  python3 scripts/data_collection_spy.py
"""

from __future__ import annotations

import gc
import json
import os
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from sqlalchemy import create_engine, Column, Integer, Float, String, Text, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import sessionmaker, declarative_base

# ==============================================================================
# Configuration
# ==============================================================================

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://flexyface:flexypass@localhost:7432/flexyface")
PROMPTS_FILE = Path(__file__).parent.parent / "benchmark_prompts.json"
OUTPUT_DIR = Path(__file__).parent.parent / "backend" / "data" / "images" / "sweep"
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/tmp/metrics_models"))

# Model config
REALVIS_MODEL_ID = "SG161222/RealVisXL_V4.0"

# Parameter ranges (expanded for research)
STEPS_RANGE = (4, 60)        # Default limits: 8-48
GUIDANCE_RANGE = (0.5, 12.0)  # Default limits: 1.0-8.0
RESOLUTION = 512
TF32_ENABLED = True
TARGET_IMAGES = 143

# ==============================================================================
# Database Models (minimal, standalone)
# ==============================================================================

Base = declarative_base()


class BenchRun(Base):
    __tablename__ = "bench_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(100), nullable=True)
    prompt = Column(Text, nullable=False)
    prompt_length = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False, default="queued")
    current_engine = Column(String(50), nullable=True)
    resolution = Column(Integer, nullable=True)
    tf32_enabled = Column(Boolean, nullable=True)
    engines_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default="now()", nullable=False)
    scoring_device = Column(String(10), nullable=True)
    scoring_load_ms = Column(Integer, nullable=True)
    scoring_inference_ms = Column(Integer, nullable=True)
    scoring_total_ms = Column(Integer, nullable=True)


class BenchRunResult(Base):
    __tablename__ = "bench_run_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), nullable=False)
    engine = Column(String(50), nullable=False)
    steps = Column(Integer, nullable=False)
    guidance = Column(Float, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    seed = Column(Integer, nullable=False)
    elapsed_ms = Column(Integer, nullable=False)
    tf32_enabled = Column(Boolean, nullable=False, default=True)
    image_path = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default="now()", nullable=False)
    clip_score = Column(Float, nullable=True)
    aesthetic_score = Column(Float, nullable=True)
    metrics_status = Column(String(20), nullable=True)
    metrics_updated_at = Column(DateTime(timezone=True), nullable=True)


# ==============================================================================
# Model Loading
# ==============================================================================

def load_realvis_pipeline() -> StableDiffusionXLPipeline:
    """Load RealVisXL pipeline with TF32 enabled."""
    print(f"Loading RealVisXL from {REALVIS_MODEL_ID}...")

    # Enable TF32 for faster compute
    if TF32_ENABLED and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        REALVIS_MODEL_ID,
        torch_dtype=torch.bfloat16,
        token=token,
    )

    # Disable safety checker
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    # Move to GPU
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    print("RealVisXL loaded successfully!")
    return pipe


def unload_pipeline(pipe: StableDiffusionXLPipeline):
    """Unload pipeline and free GPU memory."""
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ==============================================================================
# Scoring (CLIP + Aesthetic)
# ==============================================================================

def load_scorers():
    """Load CLIP and aesthetic scoring models on CPU."""
    print("Loading scoring models on CPU...")

    import open_clip

    # CLIP ViT-L/14
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    clip_model = clip_model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    # Aesthetic MLP
    aesthetic_mlp = load_aesthetic_mlp()

    print("Scoring models loaded!")
    return clip_model, preprocess, tokenizer, aesthetic_mlp


def load_aesthetic_mlp():
    """Load LAION aesthetic predictor MLP."""
    import torch.nn as nn

    class AestheticMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(768, 1024),
                nn.Dropout(0.2),
                nn.Linear(1024, 128),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.Dropout(0.1),
                nn.Linear(64, 16),
                nn.Linear(16, 1),
            )

        def forward(self, x):
            return self.layers(x)

    mlp = AestheticMLP()

    # Download weights if not present
    weights_path = MODELS_DIR / "sac+logos+ava1-l14-linearMSE.pth"
    if not weights_path.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading aesthetic weights to {weights_path}...")
        import urllib.request
        url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
        urllib.request.urlretrieve(url, weights_path)

    mlp.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    mlp.eval()
    return mlp


def score_image(img: Image.Image, prompt: str, clip_model, preprocess, tokenizer, aesthetic_mlp) -> tuple[float, float]:
    """Score an image with CLIP and aesthetic models."""
    import open_clip

    # Preprocess image
    img_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        # Get image features
        img_features = clip_model.encode_image(img_tensor)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        # CLIP score (text-image similarity)
        text_tokens = tokenizer([prompt])
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        clip_score = (img_features @ text_features.T).item()

        # Aesthetic score
        aesthetic_score = aesthetic_mlp(img_features.float()).item()

    return clip_score, aesthetic_score


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 60)
    print("RealVisXL Parameter Sweep - Data Collection")
    print("=" * 60)

    # Setup database
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    db = Session()

    # Load prompts
    print(f"Loading prompts from {PROMPTS_FILE}...")
    with open(PROMPTS_FILE) as f:
        data = json.load(f)
    prompts = [p["prompt"] for p in data["prompts"]]
    print(f"Loaded {len(prompts)} prompts")

    # Create output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Create parent BenchRun
    run_id = uuid.uuid4()
    run = BenchRun(
        id=run_id,
        session_id="data_sweep",
        prompt=f"Parameter sweep: {TARGET_IMAGES} images, steps {STEPS_RANGE}, guidance {GUIDANCE_RANGE}",
        prompt_length=0,
        status="running",
        resolution=RESOLUTION,
        tf32_enabled=TF32_ENABLED,
        scoring_device="cpu",
    )
    db.add(run)
    db.commit()
    print(f"Created BenchRun: {run_id}")

    # Load RealVisXL
    pipe = load_realvis_pipeline()

    # === GENERATION PHASE ===
    print(f"\n{'='*60}")
    print(f"Starting generation: {TARGET_IMAGES} images")
    print(f"Steps range: {STEPS_RANGE}, Guidance range: {GUIDANCE_RANGE}")
    print(f"{'='*60}\n")

    generated = 0
    cycles_needed = (TARGET_IMAGES // len(prompts)) + 1
    start_time = time.perf_counter()

    # Store prompts for scoring phase
    prompt_map = {}

    try:
        for cycle in range(cycles_needed):
            random.shuffle(prompts)
            for prompt in prompts:
                if generated >= TARGET_IMAGES:
                    break

                # Random parameters
                steps = random.randint(*STEPS_RANGE)
                guidance = round(random.uniform(*GUIDANCE_RANGE), 2)
                seed = random.randint(0, 2**31)

                # Generate
                gen_start = time.perf_counter()
                with torch.inference_mode():
                    generator = torch.Generator("cuda").manual_seed(seed)
                    result = pipe(
                        prompt=prompt,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        width=RESOLUTION,
                        height=RESOLUTION,
                        generator=generator,
                    )
                elapsed_ms = int((time.perf_counter() - gen_start) * 1000)

                # Save image
                img_id = str(uuid.uuid4())[:8]
                img_path = OUTPUT_DIR / f"sweep_{img_id}.png"
                result.images[0].save(img_path)

                # Record result
                result_id = uuid.uuid4()
                bench_result = BenchRunResult(
                    id=result_id,
                    run_id=run_id,
                    engine="realvis_xl",
                    steps=steps,
                    guidance=guidance,
                    width=RESOLUTION,
                    height=RESOLUTION,
                    seed=seed,
                    tf32_enabled=TF32_ENABLED,
                    elapsed_ms=elapsed_ms,
                    image_path=str(img_path),
                    metrics_status="pending",
                )
                db.add(bench_result)
                db.commit()

                # Store prompt for scoring
                prompt_map[result_id] = prompt

                generated += 1
                elapsed_total = time.perf_counter() - start_time
                eta_seconds = (elapsed_total / generated) * (TARGET_IMAGES - generated)
                print(f"[{generated:3d}/{TARGET_IMAGES}] steps={steps:2d}, guidance={guidance:5.2f}, {elapsed_ms:5d}ms | ETA: {eta_seconds/60:.1f}min")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving progress...")

    gen_elapsed = time.perf_counter() - start_time
    print(f"\nGeneration complete: {generated} images in {gen_elapsed/60:.1f} minutes")

    # Unload generation model
    print("Unloading RealVisXL...")
    unload_pipeline(pipe)

    # === SCORING PHASE ===
    print(f"\n{'='*60}")
    print("Starting scoring phase (CPU)...")
    print(f"{'='*60}\n")

    scoring_load_start = time.perf_counter()
    clip_model, preprocess, tokenizer, aesthetic_mlp = load_scorers()
    scoring_load_ms = int((time.perf_counter() - scoring_load_start) * 1000)
    print(f"Scoring models loaded in {scoring_load_ms}ms")

    scoring_inference_start = time.perf_counter()

    # Score all images
    results = db.query(BenchRunResult).filter(BenchRunResult.run_id == run_id).all()
    scored = 0

    for res in results:
        try:
            img = Image.open(res.image_path).convert("RGB")
            prompt = prompt_map.get(res.id, "")

            clip_score, aesthetic_score = score_image(
                img, prompt, clip_model, preprocess, tokenizer, aesthetic_mlp
            )

            res.clip_score = clip_score
            res.aesthetic_score = aesthetic_score
            res.metrics_status = "complete"
            res.metrics_updated_at = datetime.now(timezone.utc)

            scored += 1
            if scored % 50 == 0:
                db.commit()
                print(f"Scored {scored}/{len(results)}")

        except Exception as e:
            print(f"Error scoring {res.id}: {e}")
            res.metrics_status = "error"
            res.metrics_updated_at = datetime.now(timezone.utc)

    db.commit()

    scoring_inference_ms = int((time.perf_counter() - scoring_inference_start) * 1000)
    scoring_total_ms = scoring_load_ms + scoring_inference_ms

    # Update run with scoring metadata
    run.scoring_load_ms = scoring_load_ms
    run.scoring_inference_ms = scoring_inference_ms
    run.scoring_total_ms = scoring_total_ms
    run.status = "done"
    db.commit()

    total_elapsed = time.perf_counter() - start_time

    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"{'='*60}")
    print(f"Generated: {generated} images")
    print(f"Scored: {scored} images")
    print(f"Generation time: {gen_elapsed/60:.1f} min")
    print(f"Scoring time: {scoring_total_ms/1000:.1f} sec (load: {scoring_load_ms}ms, inference: {scoring_inference_ms}ms)")
    print(f"Total time: {total_elapsed/60:.1f} min")
    print(f"\nRun ID: {run_id}")
    print(f"\nQuery your data:")
    print(f"  SELECT steps, guidance, clip_score, aesthetic_score, elapsed_ms")
    print(f"  FROM bench_run_results WHERE run_id = '{run_id}';")


if __name__ == "__main__":
    main()
