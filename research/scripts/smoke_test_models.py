#!/usr/bin/env python3
"""
Quick smoke test to verify all models can load and generate.
- Runs 3 prompts (from benchmark_prompts_v2.json if present, else fallbacks)
- Uses steps=10, guidance=1.0, 512x512
- Saves a tiny CSV report and sample images under research/benchmark_results/smoke_test/
"""

import json
import os
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image, FluxPipeline, DiffusionPipeline
import pandas as pd


MODEL_REGISTRY = {
    "sdxl_turbo": {"name": "SDXL-Turbo", "repo": "stabilityai/sdxl-turbo", "guidance": 0.0, "type": "auto"},
    "flux_dev": {"name": "FLUX.1-dev", "repo": "black-forest-labs/FLUX.1-dev", "guidance": 4.5, "type": "flux"},
    "realvis_xl": {"name": "RealVisXL", "repo": "SG161222/RealVisXL_V4.0", "guidance": 1.0, "type": "auto"},
    "sd3_medium": {"name": "SD3 Medium", "repo": "stabilityai/stable-diffusion-3-medium", "guidance": 1.0, "type": "auto"},
    "sdxl": {"name": "SDXL 1.0", "repo": "stabilityai/stable-diffusion-xl-base-1.0", "guidance": 1.0, "type": "auto"},
    "hidream": {"name": "HiDream-I1", "repo": "HiDream-ai/HiDream-I1-Full", "guidance": 4.0, "type": "auto"},  # best-effort via auto
    "deepfloyd": {"name": "DeepFloyd IF", "repo": "DeepFloyd/IF-I-XL-v1.0", "guidance": 1.0, "type": "if"},
}


def load_prompts() -> list[dict]:
    prompt_path = Path(__file__).resolve().parents[1] / "data" / "benchmark_prompts_v2.json"
    if prompt_path.exists():
        with open(prompt_path, "r") as f:
            data = json.load(f)
            return data.get("prompts", [])[:3]
    return [
        {"id": 1, "prompt": "A scenic mountain lake at sunrise", "category": "landscape"},
        {"id": 2, "prompt": "Portrait of a smiling astronaut in a neon suit", "category": "people"},
        {"id": 3, "prompt": "Logo with the word NOVA in futuristic style", "category": "text"},
    ]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompts = load_prompts()
    out_dir = Path(os.environ.get("OUTPUT_ROOT", "/workspace/benchmark_results/smoke_test"))
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(exist_ok=True)

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    results = []

    for model_id, cfg in MODEL_REGISTRY.items():
        model_name = cfg["name"]
        repo = cfg["repo"]
        guidance = cfg["guidance"]
        model_type = cfg.get("type", "auto")
        if model_type == "skip":
            print(f"=== Skipping {model_name} ({repo}) for smoke test (pipeline not wired) ===")
            results.append({"model_id": model_id, "model_name": model_name, "status": "skipped", "error": "not wired"})
            continue
        print(f"=== Testing {model_name} ({repo}) ===")
        try:
            if model_type == "flux":
                pipe = FluxPipeline.from_pretrained(
                    repo,
                    torch_dtype=torch.float16,
                    cache_dir=os.environ.get("HF_HOME"),
                    token=token,
                ).to(device)
            elif model_type == "if":
                # Best-effort: load IF stage-1 pipeline
                pipe = DiffusionPipeline.from_pretrained(
                    repo,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    cache_dir=os.environ.get("HF_HOME"),
                    token=token,
                ).to(device)
            else:  # auto
                pipe = AutoPipelineForText2Image.from_pretrained(
                    repo,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    cache_dir=os.environ.get("HF_HOME"),
                    token=token,
                ).to(device)
            if hasattr(pipe, "set_progress_bar_config"):
                pipe.set_progress_bar_config(disable=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] load {model_id}: {exc}")
            results.append(
                {"model_id": model_id, "model_name": model_name, "status": "load_fail", "error": str(exc)}
            )
            continue

        for prompt_obj in prompts:
            prompt_text = prompt_obj["prompt"]
            prompt_id = prompt_obj.get("id")
            try:
                torch.cuda.empty_cache()
                start = time.perf_counter()
                gen = pipe(
                    prompt=prompt_text,
                    num_inference_steps=10,
                    guidance_scale=guidance,
                    width=512,
                    height=512,
                    generator=torch.Generator(device=device).manual_seed(42),
                )
                image = gen.images[0]
                gen_ms = (time.perf_counter() - start) * 1000.0
                filename = f"{model_id}_smoketest_prompt{prompt_id}.png"
                image_path = img_dir / filename
                image.save(image_path)
                results.append(
                    {
                        "model_id": model_id,
                        "model_name": model_name,
                        "prompt_id": prompt_id,
                        "prompt": prompt_text,
                        "status": "ok",
                        "gen_ms": gen_ms,
                        "image_path": str(image_path),
                    }
                )
                print(f"[OK] {model_id} prompt {prompt_id}: {gen_ms:.1f} ms -> {filename}")
            except Exception as exc:  # noqa: BLE001
                print(f"[FAIL] {model_id} prompt {prompt_id}: {exc}")
                results.append(
                    {
                        "model_id": model_id,
                        "model_name": model_name,
                        "prompt_id": prompt_id,
                        "prompt": prompt_text,
                        "status": "gen_fail",
                        "error": str(exc),
                    }
                )
        del pipe
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "smoke_test_results.csv", index=False)
    print(f"Smoke test complete. Results: {out_dir/'smoke_test_results.csv'}")


if __name__ == "__main__":
    main()
