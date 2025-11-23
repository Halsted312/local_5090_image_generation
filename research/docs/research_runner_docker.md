# Research Benchmark Runner Docker Skeleton

Use a separate container so the PromptPics backend keeps priority on the GPU.

## Compose snippet (aligned with docker-compose.yml)
```yaml
services:
  backend:
    # existing config, add:
    environment:
      - GPU_COORD_PATH=/gpu_coord/status.json
    volumes:
      - gpu-coord:/gpu_coord

  benchmark:
    build:
      context: .
      dockerfile: research/Dockerfile.benchmark
    env_file:
      - .env
    environment:
      - GPU_COORD_PATH=/gpu_coord/status.json
      - HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN:-}
      - HF_HUB_ENABLE_HF_TRANSFER=1
    volumes:
      - gpu-coord:/gpu_coord            # shared lease file
      - hf_cache:/root/.cache/huggingface
      - ./research:/workspace/research:ro
      - ./research/benchmark_results:/workspace/benchmark_results
    command: ["python", "research/benchmarks/efficient_benchmark_runner.py"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]

volumes:
  gpu-coord:
  hf_cache:
```

## Dockerfile stub (`research/Dockerfile.benchmark`)
```dockerfile
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
WORKDIR /workspace
COPY research/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV GPU_COORD_PATH=/gpu_coord/status.json
CMD ["python", "research/benchmarks/efficient_benchmark_runner.py"]
```

## Folder layout (research/)
- `research/benchmarks/efficient_benchmark_runner.py` — runner logic with GPU lease wrapper
- `research/docs/research_runner_docker.md` — this setup note
- `research/data/benchmark_prompts_v2.json` — 100 prompts (read-only mount)
- `research/benchmark_results/` — outputs/images/db (writeable volume)
- `research/models/` — optional local weights if not relying solely on HF cache
- `research/requirements.txt` — deps for the runner (torch/clip/aesthetic/etc.)
- `research/Dockerfile.benchmark` — container build for the runner
- `research/scripts/prefetch_models.sh` — pull all 7 models + scoring models into HF cache

## How coordination works
- Shared lease file at `/gpu_coord/status.json` (backed by `gpu-coord` volume).
- Backend wraps each generation with `owner=backend`; it waits briefly if the file says `owner=benchmark`, then proceeds and clears the file when done.
- Runner wraps each generation with `owner=benchmark`; it holds the lease only for one image, releases, and flushes CUDA (plus a 0.5s calm pause).
- Optional: poll backend `/gpu-status` for visibility (returns `busy`, `owner`, `since`).
