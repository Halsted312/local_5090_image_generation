# Research Benchmark Structure

Folders
- `research/docs/` – study notes, runner code, docker setup.
- `research/benchmark_results/` – CSV/JSON/SQLite outputs and images (mounted RW in benchmark container).
- `research/models/` – optional local copies if you decide to persist weights outside the HF cache.
- `research/scripts/` – utilities (e.g., `prefetch_models.sh`).
- `research/benchmark_prompts_v2.json` – 100 benchmark prompts (mounted RO).

Key files
- `research/docs/efficient_benchmark_runner.py` – runner with GPU lease + logging.
- `research/Dockerfile.benchmark` – image for benchmark service.
- `research/requirements.txt` – deps (diffusers/transformers/clip/aesthetic/etc.).
- `research/docs/research_runner_docker.md` – compose/mount guidance.
- `research/scripts/prefetch_models.sh` – pull all 7 models + scoring models into HF cache.

Setup steps (baseline)
1) `docker compose build benchmark` (ensures deps baked).  
2) Run `research/scripts/prefetch_models.sh` inside the benchmark image with `HUGGINGFACE_HUB_TOKEN` set. Cache stored in shared `hf_cache` volume.  
3) Ensure both backend and benchmark set `GPU_COORD_PATH=/gpu_coord/status.json` and share the `gpu-coord` volume.  
4) Start benchmark container when you want to run the study; backend keeps priority thanks to the lease + brief calm pause.  
