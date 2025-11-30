#!/usr/bin/env python3
"""
Score all images with HPS v2 and Aesthetic V2.
Runs on CPU, updates database as it goes.

Run with Docker backend STOPPED (to avoid GPU conflicts):
  docker compose stop backend
  cd /home/halsted/Python/flexy-face
  DATABASE_URL="postgresql://flexyface:flexypass@localhost:7432/flexyface" \
  python3 scripts/score_extended_metrics.py
"""

from __future__ import annotations

import os

# Set HPS v2 cache location BEFORE importing hpsv2
os.environ["HPS_ROOT"] = "/tmp/metrics_models/hpsv2"

import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://flexyface:flexypass@localhost:7432/flexyface")
MODELS_DIR = Path("/tmp/metrics_models")


class AestheticV2Scorer:
    """Score images using Aesthetic Predictor V2 Linear (ViT-L/14)."""

    def __init__(self):
        from aesthetics_predictor import AestheticsPredictorV2Linear
        from transformers import CLIPProcessor

        # Model auto-downloads from HuggingFace
        model_id = "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
        print(f"Loading Aesthetic V2 Linear from {model_id}...")
        self.model = AestheticsPredictorV2Linear.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.eval()
        print("Aesthetic V2 Linear loaded!")

    @torch.inference_mode()
    def score(self, image: Image.Image) -> float:
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        return float(outputs.logits.squeeze().item())


class HPSV2Scorer:
    """Score images using Human Preference Score v2.1."""

    def __init__(self):
        import hpsv2

        print("Loading HPS v2.1...")
        self.hps_version = "v2.1"
        # Trigger model download/load
        _ = hpsv2.score(
            Image.new("RGB", (64, 64), color="white"),
            "test",
            hps_version=self.hps_version,
        )
        print("HPS v2.1 loaded!")
        self._hpsv2 = hpsv2

    @torch.inference_mode()
    def score(self, image: Image.Image, prompt: str) -> float:
        return float(self._hpsv2.score(image, prompt, hps_version=self.hps_version))


def main():
    print("=" * 60)
    print("Extended Metrics Scoring: HPS v2 + Aesthetic V2")
    print("=" * 60)

    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    db = Session()

    # Load scorers
    load_start = time.perf_counter()
    aes_v2 = AestheticV2Scorer()
    hps_v2 = HPSV2Scorer()
    load_ms = int((time.perf_counter() - load_start) * 1000)
    print(f"Models loaded in {load_ms}ms")

    # Get all images that need scoring (where new columns are NULL)
    # Join with bench_runs to get the prompt for HPS scoring
    results = db.execute(text("""
        SELECT r.id, r.image_path, b.prompt
        FROM bench_run_results r
        JOIN bench_runs b ON r.run_id = b.id
        WHERE r.aes_v2_score IS NULL OR r.hps_v2_score IS NULL
        ORDER BY r.created_at
    """)).fetchall()

    print(f"\nScoring {len(results)} images...")
    print("=" * 60)

    inference_start = time.perf_counter()
    scored = 0
    errors = 0

    for i, (result_id, image_path, prompt) in enumerate(results):
        try:
            img = Image.open(image_path).convert("RGB")

            # Score with both models
            aes_score = aes_v2.score(img)
            hps_score = hps_v2.score(img, prompt or "")

            # Update database
            db.execute(
                text("""
                UPDATE bench_run_results
                SET aes_v2_score = :aes, hps_v2_score = :hps
                WHERE id = :id
            """),
                {"aes": aes_score, "hps": hps_score, "id": str(result_id)},
            )

            scored += 1

            if scored % 25 == 0:
                db.commit()
                elapsed = time.perf_counter() - inference_start
                rate = scored / elapsed
                eta = (len(results) - scored) / rate if rate > 0 else 0
                print(
                    f"[{scored:4d}/{len(results)}] "
                    f"aes_v2={aes_score:.2f}, hps_v2={hps_score:.4f} | "
                    f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f}min"
                )

        except Exception as e:
            print(f"Error scoring {result_id}: {e}")
            errors += 1

    db.commit()

    inference_ms = int((time.perf_counter() - inference_start) * 1000)
    total_ms = load_ms + inference_ms

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Scored: {scored} images")
    print(f"Errors: {errors}")
    print(f"Load time: {load_ms/1000:.1f}s")
    print(f"Inference time: {inference_ms/1000:.1f}s ({inference_ms/scored:.0f}ms/image)")
    print(f"Total time: {total_ms/1000:.1f}s")

    # Show sample of results
    sample = db.execute(text("""
        SELECT steps, guidance, clip_score, aesthetic_score, aes_v2_score, hps_v2_score
        FROM bench_run_results
        WHERE aes_v2_score IS NOT NULL AND hps_v2_score IS NOT NULL
        ORDER BY created_at DESC
        LIMIT 5
    """)).fetchall()

    print("\nSample results (most recent):")
    print("-" * 80)
    print(f"{'Steps':>6} {'Guidance':>8} {'CLIP':>8} {'Aes V1':>8} {'Aes V2':>8} {'HPS v2':>8}")
    print("-" * 80)
    for row in sample:
        print(
            f"{row[0]:>6} {row[1]:>8.2f} {row[2] or 0:>8.3f} "
            f"{row[3] or 0:>8.2f} {row[4] or 0:>8.2f} {row[5] or 0:>8.4f}"
        )


if __name__ == "__main__":
    main()
