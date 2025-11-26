Got it — you want:

* “What is Flux.2 exactly and how does it differ from Flux.1?”
* “Given my 5090 + Threadripper + 128 GB RAM, how should I run it (4-bit vs 8-bit etc.)?”
* “Exactly what should I tell my coding agent to do, with code stubs, including pre-downloading the right checkpoint into Docker (same pattern as your other models).”

I’ll do this in two parts:

1. **My recommendation + reasoning (for you).**
2. **A ready-to-paste AGENT PROMPT with code stubs** tailored to your FlexyFace stack.

---

## 1. Flux.2 in plain English + hardware/quantization strategy

### 1.1. What Flux.2 [dev] is

Key facts from the official sources:

* **Flux.2 [dev]** is a **32B-parameter** rectified flow transformer for image generation & editing, built by Black Forest Labs. It pairs a 24B Mistral-3 VLM text encoder with a big flow transformer and a new VAE.([Black Forest Labs][1])
* It can handle **multi-reference** editing (up to ~10 refs), text rendering, and 4MP-class resolutions with much better typography and world knowledge than Flux.1.([Black Forest Labs][1])
* The **open weights variant** you can run locally is **FLUX.2-dev**, released under the **FLUX.2-dev non-commercial license**. For commercial use you need a separate license from BFL.([Black Forest Labs][1])
* Recommended usage from their docs:

  * **Steps**: ~12–20 for previews, 28–40 for production.
  * **Guidance**: 3–5.
  * **Resolutions**: start around **1024×1024** or **1536×1024**, up to ~4MP with upscaling.([flux-2.dev][2])

So: Flux.2 [dev] is your “monster open-weight SOTA” — heavier than Flux.1, but with big wins in text, multi-reference, and 4MP detail.

---

### 1.2. Why you basically *must* quantize it on a single 5090

* 32B params at **bf16** is ~64 GB of weights *just for the transformer*, plus VAE + text encoder + activations. That’s above your 32 GB VRAM.
* BFL and Hugging Face explicitly recommend **quantized variants** (4-bit / FP8 etc.) and provide an official **4-bit bitsandbytes pipeline**, `diffusers/FLUX.2-dev-bnb-4bit`, designed for RTX 4090-class GPUs, using a **remote text encoder** to save VRAM.([GitHub][3])
* External guides show that, on a 4090/5090, a **4-bit quantized transformer + remote text encoder** is in the ~**18–20 GB VRAM** range, while fully local 4-bit TE pushes closer to ~20+ GB.([Skywork][4])

Given your **RTX 5090 (32 GB)**:

* **4-bit quantized transformer** is the **sweet spot**:

  * Fits comfortably in 32 GB with headroom for VAE, buffers, and other processes.
  * Lets you keep **steps and resolution** in a similar range to your Flux.1 setup while still being usable in web latency terms.
* **8-bit quantization** for a 32B transformer is borderline:

  * Rough math & GGUF numbers show **Q8 ~35 GB VRAM**, which is above your 32 GB budget once you add VAE + TE + overhead.([Hugging Face][5])
  * You *might* get something to limp along with heavy offloading, but it will be slower than a well-tuned 4-bit setup.
* **FP8 reference implementation** exists (NVIDIA + ComfyUI), but that’s geared to Comfy workflows rather than your existing FastAPI/diffusers backend.([Black Forest Labs][1])

**Conclusion:**

> For your FlexyFace backend + NerdSandbox and a 5090, the best first target is:
>
> * **Model**: `diffusers/FLUX.2-dev-bnb-4bit` (quantized transformer).
> * **Text encoder**: use the official **remote text encoder endpoint** from HF initially.([GitHub][3])
> * **Precision**: `torch_dtype=torch.bfloat16` (works well with modern GPUs).([Hugging Face][6])

Once that’s solid, you can experiment with:

* Local 4-bit TE, if/when an official quantized TE appears.
* FP8 / GGUF pipelines via ComfyUI if you want to benchmark them.

---

### 1.3. Target performance profile

For **“roughly on par with Flux.1 speed”** on a 5090:

* Use the 4-bit pipeline with:

  * **Resolution**: 768×768 or 896×896 for default sandbox runs.
  * **Steps**: ~20–28 (`num_inference_steps`).
  * **Guidance**: 3.5–4.5 (`guidance_scale`).([flux-2.dev][2])

The 5090’s extra compute vs 4090 should keep this within the same ballpark latency as your current Flux.1 dev runs, especially if you:

* Pin everything to CUDA,
* Use `torch.compile` where safe,
* Reuse seeds and avoid per-request re-initialization.

---

## 2. Agent prompt + code stubs (what to actually tell the agent)

Below is a **ready-to-paste prompt** you can give to your coding agent that *does* have filesystem + internet + your repo. It assumes:

* Repo: `Halsted312/flexy-face`.
* Current models: Flux.1, SD3 Medium, RealVis/HiDream etc.
* Existing download scripts: `setup_models.py`, `docker_download_models.py`.
* Model registry + router pattern: `backend/app/model_registry.py`, `backend/app/router_engine.py`, `backend/app/flux_models.py`.
* Nerd Sandbox endpoint already exists and runs multiple models for one prompt.

You can paste this whole block into the agent, or trim as needed.

---

### 🔧 AGENT PROMPT: “Add Flux.2-dev 4-bit to my FlexyFace backend and NerdSandbox”

> **Context:**
>
> You are working on the `Halsted312/flexy-face` repo.
> This repo powers a FastAPI-style backend (`backend/app/*`), a model registry (`backend/app/model_registry.py`), Flux-related wrappers (`backend/app/flux_models.py`), and a “Nerd Sandbox” endpoint that runs the same prompt through multiple models and returns both **images + timings**. Documentation for how agents should behave lives in `docs/AGENT_INSTRUCTIONS.md`, `docs/1_model_selection.md`, `docs/NERD_SANDBOX_IMPLEMENTATION.md`, etc.
>
> The user has an **RTX 5090 (32 GB VRAM)** and a **Threadripper 9970X + 128 GB RAM**. They are already running **FLUX.1** and other models locally via this repo. They now want to integrate **FLUX.2 [dev]** as a 4th model in their pipeline.

---

#### 2.1. Read these references first

1. In this repo:

   * `backend/app/flux_models.py`
   * `backend/app/model_registry.py`
   * `backend/app/router_engine.py`
   * `backend/app/schemas.py`
   * `setup_models.py` and `docker_download_models.py`
   * `docs/AGENT_INSTRUCTIONS.md`
   * `docs/NERD_SANDBOX_IMPLEMENTATION.md`
2. On the web (read them before coding):

   * Official Flux.2 blog: `https://bfl.ai/blog/flux-2` ([Black Forest Labs][1])
   * Flux2 diffusers pipeline docs: `https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux2` ([Hugging Face][6])
   * Flux2 inference repo: `https://github.com/black-forest-labs/flux2` ([GitHub][3])

Follow all existing repo instructions (especially in `docs/AGENT_INSTRUCTIONS.md`) and prefer minimal, surgical changes.

---

#### 2.2. Overall goal

Implement **FLUX.2 [dev], 4-bit quantized transformer** as a new model in this stack:

* Use **`diffusers/FLUX.2-dev-bnb-4bit`** (official 4-bit quantized transformer+pipeline).([GitHub][3])
* Use the **remote text encoder endpoint** from the Flux2 GitHub README:
  `https://remote-text-encoder-flux-2.huggingface.co/predict`, authenticated with an HF token.([GitHub][3])
* Pre-download the model into Docker using the same pattern as the other models (no online downloads at inference time).
* Register a new engine (e.g. `"flux2_dev_4bit"`) in the model registry.
* Add this engine as a **4th entry** in the Nerd Sandbox engines list.
* Keep Flux.1 as the **primary live** model for `/generate` by default.

---

#### 2.3. Step 1 – Pre-download Flux.2-dev (4-bit) into the existing models directory

1. **Identify the base models directory** used by this project (e.g. from `setup_models.py`):

   ```python
   # Example pattern – adapt to the actual code:
   from pathlib import Path
   import os

   BASE_MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models")).resolve()
   ```

2. **Add a Flux2 entry** to whatever structure enumerates models in `setup_models.py`:

   ```python
   # In setup_models.py

   FLUX2_4BIT_REPO_ID = "diffusers/FLUX.2-dev-bnb-4bit"  # official quantized repo
   FLUX2_4BIT_LOCAL_DIR = BASE_MODELS_DIR / "flux2-dev-bnb-4bit"
   ```

3. **Use `huggingface_hub.snapshot_download`** to pre-download the quantized pipeline into that dir:

   ```python
   # still in setup_models.py
   from huggingface_hub import snapshot_download

   def download_flux2_4bit():
       FLUX2_4BIT_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
       snapshot_download(
           repo_id=FLUX2_4BIT_REPO_ID,
           local_dir=str(FLUX2_4BIT_LOCAL_DIR),
           local_dir_use_symlinks=False,
           token=os.environ.get("HF_TOKEN"),  # assume HF_TOKEN already used elsewhere
       )
   ```

4. Integrate `download_flux2_4bit()` into the existing download flow:

   ```python
   def main():
       # existing downloads ...
       download_flux2_4bit()
       print("Downloaded Flux.2-dev 4-bit to", FLUX2_4BIT_LOCAL_DIR)
   ```

5. Mirror this behavior in `docker_download_models.py` (if that script is used in container builds) so the Docker image has this path populated at build/startup time.

---

#### 2.4. Step 2 – Implement a `Flux2Engine` in `backend/app/flux_models.py`

**Goal**: A new engine that:

* Loads the **local** copy of the Flux.2-dev 4-bit pipeline from `FLUX2_4BIT_LOCAL_DIR`.
* Uses the **remote text encoder** endpoint to get prompt embeddings.
* Exposes a `generate(...)` method that matches the existing Flux.1 wrapper signature so it drops into the router & NerdSandbox with minimal changes.

Add something like this (adapt names to the current abstraction, e.g. `BaseImageEngine`):

```python
# backend/app/flux_models.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import io
import os
import requests
import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from huggingface_hub import get_token

# Reuse or add a small config/dataclass if it fits your style.
@dataclass
class Flux2Config:
    model_dir: Path
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.bfloat16
    remote_text_encoder_url: str = (
        os.environ.get(
            "FLUX2_REMOTE_TEXT_ENCODER_URL",
            "https://remote-text-encoder-flux-2.huggingface.co/predict",
        )
    )


def _remote_text_encoder(
    prompts: Sequence[str],
    device: str = "cuda",
    url: Optional[str] = None,
) -> torch.Tensor:
    """
    Call the official remote text encoder for Flux.2 [dev].
    Returns a tensor of prompt embeddings on the target device.
    """

    if url is None:
        url = os.environ.get(
            "FLUX2_REMOTE_TEXT_ENCODER_URL",
            "https://remote-text-encoder-flux-2.huggingface.co/predict",
        )

    hf_token = get_token()
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        url,
        json={"prompt": list(prompts)},
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()

    # HF endpoint returns a PyTorch tensor serialized via torch.save
    prompt_embeds = torch.load(io.BytesIO(response.content))
    return prompt_embeds.to(device)


class Flux2Dev4BitEngine:
    """
    Wrapper around the diffusers Flux2Pipeline for the 4-bit quantized FLUX.2-dev model.

    This engine:
    - Loads the 4-bit quantized transformer + pipeline from a local directory
      containing the snapshot of `diffusers/FLUX.2-dev-bnb-4bit`.
    - Uses a remote text encoder to obtain prompt embeddings, as per BFL's example.
    """

    def __init__(self, config: Flux2Config):
        self.config = config
        self.device = config.device

        # We load the transformer explicitly; the snapshot contains a quantized transformer.
        repo_or_path = str(config.model_dir)

        transformer = Flux2Transformer2DModel.from_pretrained(
            repo_or_path,
            subfolder="transformer",
            torch_dtype=config.torch_dtype,
        )

        self.pipe = Flux2Pipeline.from_pretrained(
            repo_or_path,
            transformer=transformer,
            text_encoder=None,  # we supply prompt_embeds directly
            torch_dtype=config.torch_dtype,
        ).to(self.device)

        # Optional: enable memory/time optimizations
        self.pipe.enable_model_cpu_offload = False
        self.pipe.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        seed: int = 0,
        num_inference_steps: int = 28,
        guidance_scale: float = 4.0,
        width: int = 768,
        height: int = 768,
    ):
        generator = torch.Generator(device=self.device).manual_seed(seed)

        prompt_embeds = _remote_text_encoder(
            prompts=[prompt],
            device=self.device,
            url=self.config.remote_text_encoder_url,
        )

        result = self.pipe(
            prompt_embeds=prompt_embeds,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        )

        image = result.images[0]
        return image
```

**Notes for the agent:**

* Copy the **behavior** and signature from your existing Flux.1 wrapper so that the router can treat them the same way.
* If the repo already has a `BaseImageEngine` class or `ImageBackend` protocol, make `Flux2Dev4BitEngine` implement that instead of defining a fresh interface.
* Keep default `num_inference_steps` / `guidance_scale` in the ranges recommended for Flux.2 dev: 20–28 steps for speed, guidance ~3–5.([flux-2.dev][2])

---

#### 2.5. Step 3 – Wire it into the model registry

In `backend/app/model_registry.py`, register this new engine using whatever pattern you already use.

**Example pattern:**

```python
# backend/app/model_registry.py

from pathlib import Path
from .flux_models import Flux2Config, Flux2Dev4BitEngine

# Example: base models dir imported or recomputed here
BASE_MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models")).resolve()
FLUX2_4BIT_LOCAL_DIR = BASE_MODELS_DIR / "flux2-dev-bnb-4bit"

def build_flux2_dev_4bit() -> Flux2Dev4BitEngine:
    cfg = Flux2Config(model_dir=FLUX2_4BIT_LOCAL_DIR)
    return Flux2Dev4BitEngine(cfg)

MODEL_REGISTRY = {
    # existing entries, e.g:
    # "flux1_dev": build_flux1_dev,
    # "sd3_medium": build_sd3_medium,
    # ...
    "flux2_dev_4bit": build_flux2_dev_4bit,
}
```

If your registry uses dataclasses or config objects instead of functions, adapt the stub accordingly.

---

#### 2.6. Step 4 – Extend schemas and router

1. In `backend/app/schemas.py`, add `"flux2_dev_4bit"` to the engine enum / Literal:

   ```python
   # Example
   from enum import Enum

   class EngineName(str, Enum):
       flux1_dev = "flux1_dev"
       sd3_medium = "sd3_medium"
       realvis_xl = "realvis_xl"
       # ...
       flux2_dev_4bit = "flux2_dev_4bit"
   ```

2. In `backend/app/router_engine.py`, map the new engine name to the registry:

   ```python
   from .model_registry import MODEL_REGISTRY

   def generate_image(request: PromptRequest) -> ImageResult:
       engine_name = request.engine or "flux1_dev"
       engine = MODEL_REGISTRY[engine_name]()
       return engine.generate(
           prompt=request.prompt,
           seed=request.seed,
           num_inference_steps=request.steps,
           guidance_scale=request.guidance,
           width=request.width,
           height=request.height,
       )
   ```

   If your router already works this way, ensure that it **does not special-case** Flux.1/Flux.2 beyond engine names.

---

#### 2.7. Step 5 – Add Flux.2 to NerdSandbox

Find the NerdSandbox implementation (look for `NERD_SANDBOX_IMPLEMENTATION` doc and the corresponding endpoint in `backend/app/main.py` / `router_engine.py`). You likely have something like:

```python
NERD_SANDBOX_ENGINES = [
    EngineName.flux1_dev,
    EngineName.sd3_medium,
    EngineName.realvis_xl,
]
```

Update it to:

```python
NERD_SANDBOX_ENGINES = [
    EngineName.flux1_dev,
    EngineName.sd3_medium,
    EngineName.realvis_xl,
    EngineName.flux2_dev_4bit,
]
```

And ensure the loop is generic:

```python
def run_nerd_sandbox(req: PromptRequest) -> NerdSandboxResponse:
    results: list[NerdSandboxModelResult] = []
    for engine_name in NERD_SANDBOX_ENGINES:
        t0 = time.perf_counter()
        engine = MODEL_REGISTRY[engine_name]()
        image = engine.generate(
            prompt=req.prompt,
            seed=req.seed,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance,
            width=req.width,
            height=req.height,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # encode image to whatever format the API uses (base64/url)
        encoded = encode_image(image)

        results.append(
            NerdSandboxModelResult(
                engine=str(engine_name.value),
                latency_ms=elapsed_ms,
                image=encoded,
            )
        )
    return NerdSandboxResponse(results=results)
```

Now NerdSandbox will return 4 images + 4 timings per prompt, including Flux.2.

---

#### 2.8. Step 6 – Tests & example API calls

1. **Unit/integration smoke test** for Flux.2:

   ```python
   def test_flux2_dev_4bit_smoke():
       from backend.app.flux_models import Flux2Config, Flux2Dev4BitEngine
       from pathlib import Path
       import os

       base_dir = Path(os.environ.get("MODELS_DIR", "/models"))
       model_dir = base_dir / "flux2-dev-bnb-4bit"

       engine = Flux2Dev4BitEngine(
           Flux2Config(model_dir=model_dir, device="cuda")
       )
       img = engine.generate(
           prompt="A simple line drawing of a smiling face",
           seed=123,
           num_inference_steps=12,
           guidance_scale=3.0,
           width=512,
           height=512,
       )
       assert img is not None
   ```

2. **Example `/generate` call using Flux.2**:

   ```json
   POST /generate
   {
     "prompt": "ultra realistic portrait of a corgi astronaut with legible mission patch text 'PROMPTPICS', studio lighting",
     "engine": "flux2_dev_4bit",
     "seed": 42,
     "steps": 24,
     "guidance": 4.0,
     "width": 768,
     "height": 768
   }
   ```

3. **Example NerdSandbox call** (no explicit engine – it will run all 4):

   ```json
   POST /nerd-sandbox
   {
     "prompt": "infographic-style poster that says 'PROMPTPICS PRESENTS' in big text, plus a realistic photo panel and a cartoon panel",
     "seed": 123,
     "steps": 24,
     "guidance": 4.0,
     "width": 768,
     "height": 768
   }
   ```

   Response: `results` array should now have **4 entries**, one of which is `engine: "flux2_dev_4bit"`.

---

#### 2.9. Notes on quantization choices for future experiments

* **Fastest config for today** (recommended baseline):

  * `diffusers/FLUX.2-dev-bnb-4bit` + remote text encoder, as implemented above.([GitHub][3])
* **Higher fidelity / slower configs** to try later:

  * Higher resolution (1024×1024, 1536×1024) and more steps (28–40).([flux-2.dev][2])
  * FP8 / GGUF pipelines via ComfyUI if you want to build a separate path for benchmarking, but *keep this separate* from the main REST API until stable.([Black Forest Labs][1])
* **License**: Flux.2-dev is open weights but **non-commercial**. If this backend is used in a commercial PromptPics product, make sure to follow the **FLUX.2-dev Non-Commercial License** and obtain a commercial license from Black Forest Labs if needed.([Black Forest Labs][1])

---


[1]: https://bfl.ai/blog/flux-2 "FLUX.2: Frontier Visual Intelligence | Black Forest Labs"
[2]: https://flux-2.dev/?utm_source=chatgpt.com "FLUX 2 Dev | FLUX.2-dev Guide"
[3]: https://github.com/black-forest-labs/flux2 "GitHub - black-forest-labs/flux2: Official inference repo for FLUX.2 models"
[4]: https://skywork.ai/blog/ai-agent/flux-2-dev-2025-a-complete-beginners-handbook-to-get-started/?utm_source=chatgpt.com "FLUX.2 Dev 2025: A Complete Beginner's Handbook to Get Started - Skywork ai"
[5]: https://huggingface.co/city96/FLUX.2-dev-gguf?utm_source=chatgpt.com "city96/FLUX.2-dev-gguf · Hugging Face"
[6]: https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux2 "Flux2"
