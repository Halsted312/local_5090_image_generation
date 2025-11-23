# Vision Model Study Notes – PromptPics Backend

Date: 2025-11-22  
Target GPU: NVIDIA 5090 (TF32-capable, ample VRAM)  
Source files: `backend/app/flux_models.py`, `backend/app/main.py`, `backend/app/router_engine.py`, `backend/app/model_registry.py`, `docs/hidream-integration_2025-11-22_17-23-48.md`

## Model Inventory (current backend)

| ID            | Display Name               | Role / Strengths                                   | Default Res | Default Steps / Guidance (suggested) | Memory Notes                           |
|---------------|---------------------------|----------------------------------------------------|-------------|---------------------------------------|----------------------------------------|
| `flux_dev`    | FLUX.1-dev                | General-purpose, stylized, landscapes, strong PF   | 768×768 cap | 24–28 steps, guidance 4.0–6.0         | Fits GPU; can stay resident            |
| `realvis_xl`  | RealVisXL V4.0            | Photoreal portraits/people                        | 768×768 cap | 24–28 steps, guidance 4.5–6.5         | Fits GPU; can stay resident            |
| `sd3_medium`  | Stable Diffusion 3 Medium | Complex prompts, posters/UI, better typography     | 768×768 cap | 28–32 steps, guidance 5.5–7.0         | Larger; keep if VRAM allows            |
| `logo_sdxl`   | HiDream I1 (Text & Logos) | 17B HiDream-I1-Full + Llama-3.1-8B text encoder; superior text/logo rendering | 768×768 cap | 18–24 steps, guidance 3.5–5.0         | Very large; unload others before load  |

Resolution clamp: backend now caps width/height to 768 (`MAX_IMAGE_SIDE` env, default 768).  
TF32: enabled at startup (`torch.backends.cuda.matmul.allow_tf32 = True`, `cudnn.allow_tf32 = True`).

## Load / Unload Strategy

- Global generation lock: requests queue to avoid VRAM contention.
- Keep last-used pipeline resident; unload others only when loading HiDream or on OOM. Avoid clearing CUDA between same-model runs; clear before/after swaps.
- HiDream (`logo_sdxl`): unload other models first, load with fp16 text encoder + sequential CPU offload fallback; safety checker disabled to avoid black outputs.
- Router LLM: small instruct model (`Qwen/Qwen2.5-1.5B-Instruct`) for routing; same class used for NSFW prompt rewrites.

## Safety / Fallback Flow

- NSFW flag and black-image detector: if `nsfw_content_detected` or the image is essentially black, rewrite prompt via small LLM (“make it safe, disambiguate drug-ish terms”) and retry.  
- If HiDream still returns black, fall back to `flux_dev` to return a usable image.

## Recommended Frontend Defaults

If only one default is allowed, use: steps 24, guidance 4.5 (backend clamps res to 768).  
If per-model defaults become possible:
- flux_dev: 24–28 steps, guidance 4.0–6.0
- realvis_xl: 24–28 steps, guidance 4.5–6.5
- sd3_medium: 28–32 steps, guidance 5.5–7.0
- logo_sdxl (HiDream): 18–24 steps, guidance 3.5–5.0, keep 768×768

## Code Stubs (reference)

Enable TF32 (already in `main.py`):
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

Resolution clamp (in `_process_generation`):
```python
if request.width and request.width > MAX_IMAGE_SIDE:
    request.width = MAX_IMAGE_SIDE
if request.height and request.height > MAX_IMAGE_SIDE:
    request.height = MAX_IMAGE_SIDE
```

HiDream load (simplified):
```python
if "HiDream" in LOGO_SDXL_MODEL_ID:
    unload_all_models_except("logo")
    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(HIDREAM_TEXT_ENCODER_ID, token=token)
    text_encoder_4 = LlamaForCausalLM.from_pretrained(HIDREAM_TEXT_ENCODER_ID, torch_dtype=torch.float16, device_map="balanced", low_cpu_mem_usage=True)
    pipe = HiDreamImagePipeline.from_pretrained(LOGO_SDXL_MODEL_ID, tokenizer_4=tokenizer_4, text_encoder_4=text_encoder_4, torch_dtype=torch.float16, low_cpu_mem_usage=True, token=token)
    pipe.safety_checker = None
    pipe.enable_sequential_cpu_offload()  # fallback to enable_model_cpu_offload if needed
    pipe.enable_attention_slicing(); pipe.enable_vae_slicing()
```

Generation fallback (simplified):
```python
result = pipe(...)
image = result.images[0]
if nsfw or is_black(image):
    rewritten = rewrite_prompt_safe(prompt)
    result = pipe(prompt=rewritten, ...)
    image = result.images[0]
    if is_black(image) and model_id == "logo_sdxl":
        # fall back to flux_dev
        pipe = get_text_pipeline()
        result = pipe(prompt=rewritten, ...)
        image = result.images[0]
```

## Env Vars of Interest
- `MAX_IMAGE_SIDE` (default 768) – resolution cap.
- `REWRITE_LLM_ID` / `ROUTER_LLM_ID` – small LLM for routing/rewrites (default Qwen2.5-1.5B-Instruct).
- `LOGO_SDXL_MODEL_ID` – points to HiDream-I1-Full.
- TF32 is hard-enabled in code (no env needed).

## Open Questions / Next Steps for Study
- Benchmark per-model latency at 768 with TF32 vs offload/no-offload.
- Evaluate quality vs steps/guidance presets per model for logos/text vs photos.
- Consider keeping two pipelines resident (e.g., flux_dev + HiDream) if VRAM allows; otherwise keep current unload-on-swap behavior.
- Monitor NSFW/black incidence after rewrite to tune the rewrite prompt further.
