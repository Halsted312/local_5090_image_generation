# Hugging Face Download Checklist (Benchmark)

Use this to confirm access before long runs.

1) **Token**: Set `HUGGINGFACE_HUB_TOKEN` in your `.env` (used by backend + benchmark services).  
2) **License gates**: Make sure the account tied to the token has accepted any gated models:
   - `stabilityai/stable-diffusion-3-medium` (may require org access)
   - `DeepFloyd/IF-I-XL-v1.0` (requires license acceptance)
   - `HiDream-ai/HiDream-I1-Full` (check vendor terms)
   - Others are typically public, but verify.
3) **Auth check**: Run `research/scripts/check_hf_access.sh` inside the benchmark container (or host with same env). Fails fast if a gate is not accepted.
4) **Prefetch**: Run `research/scripts/prefetch_models.sh` to populate `hf_cache` volume so benchmark runs do not pull at runtime.
5) **Cache location**: Shared HF cache volume `hf_cache` is mounted at `/root/.cache/huggingface` in backend + benchmark.

Commands (inside container or host with token in env):
```bash
./research/scripts/check_hf_access.sh
./research/scripts/prefetch_models.sh
```
