#!/usr/bin/env python3
"""
Backfill script for CLIP and aesthetic scores.

Processes existing benchmark images that don't have scores yet.
Runs entirely on CPU to avoid GPU contention.

Usage:
    METRICS_CPU_THREADS=26 python scripts/backfill_metrics_cpu.py

Environment variables:
    METRICS_CPU_THREADS: Number of CPU threads for PyTorch (default: 26)
    MODELS_DIR: Directory for model weights (default: /models)
    DATABASE_URL: PostgreSQL connection string
    BATCH_SIZE: Number of images to process per batch (default: 64)
    DRY_RUN: Set to "1" to preview without updating database
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Set thread count BEFORE importing torch
NUM_THREADS = int(os.environ.get("METRICS_CPU_THREADS", "26"))
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)

import torch
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(2)

from PIL import Image

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.database import SessionLocal
from app.models import BenchRunResult, BenchRun
from app.metrics_models import MetricsConfig, ClipScorer, AestheticScorer


def main():
    models_dir = Path(os.environ.get("MODELS_DIR", "/models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(os.environ.get("BATCH_SIZE", "64"))
    dry_run = os.environ.get("DRY_RUN", "0") == "1"

    print("=" * 60)
    print("Metrics Backfill Script")
    print("=" * 60)
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"Models directory: {models_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Dry run: {dry_run}")
    print()

    # Load models
    print("Loading models...")
    cfg = MetricsConfig(models_dir=models_dir, device="cpu")

    load_start = time.perf_counter()
    clip = ClipScorer(cfg)
    print(f"  CLIP loaded in {time.perf_counter() - load_start:.2f}s")

    load_start = time.perf_counter()
    aest = AestheticScorer(cfg)
    print(f"  Aesthetic loaded in {time.perf_counter() - load_start:.2f}s")
    print()

    # Connect to database
    session = SessionLocal()

    # Count total pending
    total_pending = (
        session.query(BenchRunResult)
        .filter(BenchRunResult.clip_score.is_(None))
        .count()
    )
    print(f"Total images to process: {total_pending}")
    print()

    if total_pending == 0:
        print("No images to process. Done!")
        session.close()
        return

    processed = 0
    errors = 0
    start_time = time.perf_counter()

    while True:
        # Fetch batch of unprocessed results (skip errors to avoid infinite loop)
        rows = (
            session.query(BenchRunResult)
            .filter(BenchRunResult.clip_score.is_(None))
            .filter(
                (BenchRunResult.metrics_status.is_(None)) |
                (BenchRunResult.metrics_status != "error")
            )
            .order_by(BenchRunResult.created_at.asc())
            .limit(batch_size)
            .all()
        )

        if not rows:
            break

        batch_start = time.perf_counter()

        for row in rows:
            try:
                # Load image - map Docker path to host path
                img_path_str = row.image_path

                # Map Docker container path to host path
                if img_path_str.startswith("/app/"):
                    # Docker: /app/data/images/bench/... -> Host: ./backend/data/images/bench/...
                    host_path = img_path_str.replace(
                        "/app/data/images/bench/",
                        str(Path(__file__).parent.parent / "backend" / "data" / "images" / "bench") + "/"
                    )
                    img_path = Path(host_path)
                else:
                    img_path = Path(img_path_str)

                if not img_path.exists():
                    raise FileNotFoundError(f"Image not found: {img_path}")

                img = Image.open(img_path).convert("RGB")

                # Get prompt from associated run
                run = session.query(BenchRun).filter(BenchRun.id == row.run_id).first()
                prompt = run.prompt if run else ""

                # Compute scores
                clip_score = clip.score(img, prompt)
                aest_score = aest.score(img)

                if not dry_run:
                    row.clip_score = clip_score
                    row.aesthetic_score = aest_score
                    row.metrics_status = "complete"
                    row.metrics_updated_at = datetime.now(timezone.utc)

                processed += 1

            except Exception as e:
                errors += 1
                print(f"  Error processing {row.id}: {e}")

                if not dry_run:
                    row.metrics_status = "error"
                    row.metrics_updated_at = datetime.now(timezone.utc)

        if not dry_run:
            session.commit()

        batch_time = time.perf_counter() - batch_start
        elapsed = time.perf_counter() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = (total_pending - processed) / rate if rate > 0 else 0

        print(
            f"Batch: {len(rows)} images in {batch_time:.1f}s | "
            f"Total: {processed}/{total_pending} | "
            f"Rate: {rate:.1f} img/s | "
            f"ETA: {remaining:.0f}s"
        )

    session.close()

    elapsed = time.perf_counter() - start_time
    print()
    print("=" * 60)
    print("Backfill Complete")
    print("=" * 60)
    print(f"Processed: {processed}")
    print(f"Errors: {errors}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Rate: {processed / elapsed:.2f} images/second")


if __name__ == "__main__":
    main()
