#!/usr/bin/env python3
"""
CPU-based metrics benchmark script.

Measures per-image scoring time for CLIP and aesthetic models
at different resolutions.

Usage:
    METRICS_CPU_THREADS=26 python scripts/benchmark_metrics_cpu.py

Environment variables:
    METRICS_CPU_THREADS: Number of CPU threads for PyTorch (default: 26)
    BENCHMARK_IMAGE: Path to test image (default: uses a generated test image)
    MODELS_DIR: Directory for model weights (default: /models)
"""

import os
import sys
import time

# Set thread count BEFORE importing torch
NUM_THREADS = int(os.environ.get("METRICS_CPU_THREADS", "26"))
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)

import torch
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(2)

from pathlib import Path
from PIL import Image
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.metrics_models import MetricsConfig, ClipScorer, AestheticScorer


def create_test_image(size: int = 1024) -> Image.Image:
    """Create a synthetic test image if none provided."""
    # Create a gradient image with some noise
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            arr[i, j] = [
                int(255 * i / size),
                int(255 * j / size),
                int(128 + 127 * np.sin(i * j / 10000)),
            ]
    # Add noise
    noise = np.random.randint(0, 30, (size, size, 3), dtype=np.uint8)
    arr = np.clip(arr.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def benchmark_for_size(
    img: Image.Image,
    prompt: str,
    clip_scorer: ClipScorer,
    aest_scorer: AestheticScorer,
    size: int,
    repeats: int = 5,
) -> dict:
    """Benchmark scoring at a specific resolution."""
    resized = img.resize((size, size), Image.LANCZOS).convert("RGB")

    # Warm up
    _ = clip_scorer.score(resized, prompt)
    _ = aest_scorer.score(resized)

    # Benchmark CLIP
    clip_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        clip_score = clip_scorer.score(resized, prompt)
        clip_times.append(time.perf_counter() - start)

    # Benchmark Aesthetic
    aest_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        aest_score = aest_scorer.score(resized)
        aest_times.append(time.perf_counter() - start)

    # Combined benchmark
    combined_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = clip_scorer.score(resized, prompt)
        _ = aest_scorer.score(resized)
        combined_times.append(time.perf_counter() - start)

    return {
        "size": size,
        "clip_score": clip_score,
        "aest_score": aest_score,
        "clip_avg_ms": np.mean(clip_times) * 1000,
        "clip_std_ms": np.std(clip_times) * 1000,
        "aest_avg_ms": np.mean(aest_times) * 1000,
        "aest_std_ms": np.std(aest_times) * 1000,
        "combined_avg_ms": np.mean(combined_times) * 1000,
        "combined_std_ms": np.std(combined_times) * 1000,
    }


def main():
    models_dir = Path(os.environ.get("MODELS_DIR", "/models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    cfg = MetricsConfig(models_dir=models_dir, device="cpu")

    print(f"CPU Metrics Benchmark")
    print(f"{'=' * 60}")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"Models directory: {models_dir}")
    print()

    print("Loading CLIP model (ViT-L/14)...")
    load_start = time.perf_counter()
    clip = ClipScorer(cfg)
    clip_load_time = time.perf_counter() - load_start
    print(f"  CLIP loaded in {clip_load_time:.2f}s")

    print("Loading Aesthetic model...")
    load_start = time.perf_counter()
    aest = AestheticScorer(cfg)
    aest_load_time = time.perf_counter() - load_start
    print(f"  Aesthetic loaded in {aest_load_time:.2f}s")
    print()

    # Load or create test image
    img_path = os.environ.get("BENCHMARK_IMAGE")
    if img_path and Path(img_path).exists():
        print(f"Using image: {img_path}")
        img = Image.open(img_path).convert("RGB")
    else:
        print("Using synthetic test image (1024x1024)")
        img = create_test_image(1024)

    prompt = "A photorealistic high-quality image with fine details"
    print(f"Test prompt: '{prompt}'")
    print()

    # Run benchmarks
    print(f"{'Size':<10} {'CLIP (ms)':<15} {'Aesthetic (ms)':<18} {'Combined (ms)':<18} {'Scores'}")
    print("-" * 80)

    for size in (224, 512, 768, 1024):
        results = benchmark_for_size(img, prompt, clip, aest, size, repeats=5)
        print(
            f"{size}x{size:<6} "
            f"{results['clip_avg_ms']:>6.1f} +/- {results['clip_std_ms']:>4.1f}   "
            f"{results['aest_avg_ms']:>6.1f} +/- {results['aest_std_ms']:>4.1f}     "
            f"{results['combined_avg_ms']:>6.1f} +/- {results['combined_std_ms']:>4.1f}     "
            f"CLIP={results['clip_score']:.3f} Aest={results['aest_score']:.2f}"
        )

    print()
    print("Notes:")
    print("  - CLIP ViT-L/14 internally resizes to 224x224, so input size has minimal impact")
    print("  - Combined time may be less than sum due to caching effects")
    print("  - Aesthetic scorer reuses CLIP embeddings, so it's faster after CLIP")


if __name__ == "__main__":
    main()
