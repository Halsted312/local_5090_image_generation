#!/usr/bin/env python3
"""
Score all images with 4 models on GPU, one model at a time.
Fast and simple - loads each model, scores all images, clears VRAM, next model.

Run with Docker backend STOPPED:
  docker compose stop backend
  cd /home/halsted/Python/flexy-face
  DATABASE_URL="postgresql://flexyface:flexypass@localhost:7432/flexyface" \
  python3 scripts/score_all_gpu.py
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path

os.environ["HPS_ROOT"] = "/tmp/metrics_models/hpsv2"

import torch
from PIL import Image
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://flexyface:flexypass@localhost:7432/flexyface")
MODELS_DIR = Path("/tmp/metrics_models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 50  # Commit every N images


def clear_gpu():
    """Clear GPU memory between models."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# Model 1 & 2: CLIP + Aesthetic V1 (uses same CLIP model)
# =============================================================================

def score_clip_and_aesthetic_v1(db):
    """Score with CLIP and Aesthetic V1 on GPU."""
    import open_clip
    import torch.nn as nn

    print("\n" + "=" * 60)
    print("MODEL 1 & 2: CLIP + Aesthetic V1 (GPU)")
    print("=" * 60)

    # Get images missing either score
    results = db.execute(text("""
        SELECT r.id, r.image_path, b.prompt
        FROM bench_run_results r
        JOIN bench_runs b ON r.run_id = b.id
        WHERE r.clip_score IS NULL OR r.aesthetic_score IS NULL
        ORDER BY r.created_at
    """)).fetchall()

    if not results:
        print("No images need CLIP/Aesthetic V1 scoring")
        return

    print(f"Scoring {len(results)} images...")

    # Load CLIP
    load_start = time.perf_counter()
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    clip_model = clip_model.to(DEVICE).eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    # Load Aesthetic MLP
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

    aesthetic_mlp = AestheticMLP()
    weights_path = MODELS_DIR / "sac+logos+ava1-l14-linearMSE.pth"
    if not weights_path.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading aesthetic weights...")
        import urllib.request
        url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
        urllib.request.urlretrieve(url, weights_path)

    aesthetic_mlp.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    aesthetic_mlp = aesthetic_mlp.to(DEVICE).eval()

    load_ms = int((time.perf_counter() - load_start) * 1000)
    print(f"Models loaded in {load_ms}ms")

    # Score
    inference_start = time.perf_counter()
    scored = 0

    for result_id, image_path, prompt in results:
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

            with torch.inference_mode():
                img_features = clip_model.encode_image(img_tensor)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)

                # CLIP score
                text_tokens = tokenizer([prompt or ""]).to(DEVICE)
                text_features = clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                clip_score = (img_features @ text_features.T).item()

                # Aesthetic score
                aesthetic_score = aesthetic_mlp(img_features.float()).item()

            db.execute(
                text("UPDATE bench_run_results SET clip_score = :clip, aesthetic_score = :aes WHERE id = :id"),
                {"clip": clip_score, "aes": aesthetic_score, "id": str(result_id)},
            )
            scored += 1

            if scored % BATCH_SIZE == 0:
                db.commit()
                rate = scored / (time.perf_counter() - inference_start)
                print(f"[{scored:4d}/{len(results)}] clip={clip_score:.3f}, aes={aesthetic_score:.2f} | {rate:.1f}/s")

        except Exception as e:
            print(f"Error: {result_id}: {e}")

    db.commit()
    inference_ms = int((time.perf_counter() - inference_start) * 1000)
    print(f"Done: {scored} images in {inference_ms/1000:.1f}s ({inference_ms/max(scored,1):.0f}ms/img)")

    # Cleanup
    del clip_model, aesthetic_mlp, preprocess, tokenizer
    clear_gpu()


# =============================================================================
# Model 3: Aesthetic V2
# =============================================================================

def score_aesthetic_v2(db):
    """Score with Aesthetic V2 on GPU."""
    from aesthetics_predictor import AestheticsPredictorV2Linear
    from transformers import CLIPProcessor

    print("\n" + "=" * 60)
    print("MODEL 3: Aesthetic V2 (GPU)")
    print("=" * 60)

    # Get images missing score
    results = db.execute(text("""
        SELECT id, image_path FROM bench_run_results
        WHERE aes_v2_score IS NULL
        ORDER BY created_at
    """)).fetchall()

    if not results:
        print("No images need Aesthetic V2 scoring")
        return

    print(f"Scoring {len(results)} images...")

    # Load model
    load_start = time.perf_counter()
    model_id = "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
    model = AestheticsPredictorV2Linear.from_pretrained(model_id).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    load_ms = int((time.perf_counter() - load_start) * 1000)
    print(f"Model loaded in {load_ms}ms")

    # Score
    inference_start = time.perf_counter()
    scored = 0

    for result_id, image_path in results:
        try:
            img = Image.open(image_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = model(**inputs)
                score = outputs.logits.squeeze().item()

            db.execute(
                text("UPDATE bench_run_results SET aes_v2_score = :score WHERE id = :id"),
                {"score": score, "id": str(result_id)},
            )
            scored += 1

            if scored % BATCH_SIZE == 0:
                db.commit()
                rate = scored / (time.perf_counter() - inference_start)
                print(f"[{scored:4d}/{len(results)}] aes_v2={score:.2f} | {rate:.1f}/s")

        except Exception as e:
            print(f"Error: {result_id}: {e}")

    db.commit()
    inference_ms = int((time.perf_counter() - inference_start) * 1000)
    print(f"Done: {scored} images in {inference_ms/1000:.1f}s ({inference_ms/max(scored,1):.0f}ms/img)")

    # Cleanup
    del model, processor
    clear_gpu()


# =============================================================================
# Model 4: HPS v2
# =============================================================================

def score_hps_v2(db):
    """Score with HPS v2 on GPU."""
    import hpsv2

    print("\n" + "=" * 60)
    print("MODEL 4: HPS v2.1 (GPU)")
    print("=" * 60)

    # Get images missing score
    results = db.execute(text("""
        SELECT r.id, r.image_path, b.prompt
        FROM bench_run_results r
        JOIN bench_runs b ON r.run_id = b.id
        WHERE r.hps_v2_score IS NULL
        ORDER BY r.created_at
    """)).fetchall()

    if not results:
        print("No images need HPS v2 scoring")
        return

    print(f"Scoring {len(results)} images...")

    # Warm up model (downloads on first call)
    load_start = time.perf_counter()
    _ = hpsv2.score(Image.new("RGB", (64, 64), color="white"), "test", hps_version="v2.1")
    load_ms = int((time.perf_counter() - load_start) * 1000)
    print(f"Model loaded in {load_ms}ms")

    # Score
    inference_start = time.perf_counter()
    scored = 0

    for result_id, image_path, prompt in results:
        try:
            img = Image.open(image_path).convert("RGB")

            with torch.inference_mode():
                result = hpsv2.score(img, prompt or "", hps_version="v2.1")
                # Handle both list and scalar returns
                score = float(result[0]) if isinstance(result, list) else float(result)

            db.execute(
                text("UPDATE bench_run_results SET hps_v2_score = :score WHERE id = :id"),
                {"score": score, "id": str(result_id)},
            )
            scored += 1

            if scored % BATCH_SIZE == 0:
                db.commit()
                rate = scored / (time.perf_counter() - inference_start)
                print(f"[{scored:4d}/{len(results)}] hps_v2={score:.4f} | {rate:.1f}/s")

        except Exception as e:
            print(f"Error: {result_id}: {e}")

    db.commit()
    inference_ms = int((time.perf_counter() - inference_start) * 1000)
    print(f"Done: {scored} images in {inference_ms/1000:.1f}s ({inference_ms/max(scored,1):.0f}ms/img)")

    clear_gpu()


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("4-Model GPU Scoring")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    db = Session()

    # Check current status
    status = db.execute(text("""
        SELECT COUNT(*) as total,
               COUNT(clip_score) as clip,
               COUNT(aesthetic_score) as aes_v1,
               COUNT(aes_v2_score) as aes_v2,
               COUNT(hps_v2_score) as hps_v2
        FROM bench_run_results
    """)).fetchone()

    print(f"\nCurrent status:")
    print(f"  Total images: {status[0]}")
    print(f"  CLIP scores: {status[1]}")
    print(f"  Aesthetic V1: {status[2]}")
    print(f"  Aesthetic V2: {status[3]}")
    print(f"  HPS v2: {status[4]}")

    total_start = time.perf_counter()

    # Run each model in sequence
    score_clip_and_aesthetic_v1(db)
    score_aesthetic_v2(db)
    score_hps_v2(db)

    total_sec = time.perf_counter() - total_start

    # Final status
    final = db.execute(text("""
        SELECT COUNT(*) as total,
               COUNT(clip_score) as clip,
               COUNT(aesthetic_score) as aes_v1,
               COUNT(aes_v2_score) as aes_v2,
               COUNT(hps_v2_score) as hps_v2
        FROM bench_run_results
    """)).fetchone()

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Total time: {total_sec/60:.1f} minutes")
    print(f"\nFinal status:")
    print(f"  Total images: {final[0]}")
    print(f"  CLIP scores: {final[1]}")
    print(f"  Aesthetic V1: {final[2]}")
    print(f"  Aesthetic V2: {final[3]}")
    print(f"  HPS v2: {final[4]}")

    # Sample results
    sample = db.execute(text("""
        SELECT steps, guidance, clip_score, aesthetic_score, aes_v2_score, hps_v2_score
        FROM bench_run_results
        WHERE clip_score IS NOT NULL
          AND aesthetic_score IS NOT NULL
          AND aes_v2_score IS NOT NULL
          AND hps_v2_score IS NOT NULL
        ORDER BY created_at DESC
        LIMIT 5
    """)).fetchall()

    if sample:
        print("\nSample results (most recent):")
        print("-" * 80)
        print(f"{'Steps':>6} {'Guidance':>8} {'CLIP':>8} {'Aes V1':>8} {'Aes V2':>8} {'HPS v2':>8}")
        print("-" * 80)
        for row in sample:
            print(f"{row[0]:>6} {row[1]:>8.2f} {row[2] or 0:>8.3f} {row[3] or 0:>8.2f} {row[4] or 0:>8.2f} {row[5] or 0:>8.4f}")


if __name__ == "__main__":
    main()
