Right now your backend has the *plumbing* for a router (`RoutingDecision`, `router.py`, etc.), but the “brain” was still a bit hand‑wavy. Let’s fix that and actually spell out:

* **Exactly what each model is for**
* **Exactly how we score them per prompt**
* **Exactly how a small LLM (your “fast model”) should reason and output JSON**

I’ll show both:

1. A **rule‑based scoring function** you can drop straight into Python (fast, deterministic).
2. A **router LLM prompt** that encodes the same logic, so an 8B model like Llama‑3‑8B can do the reasoning in natural language and still output structured JSON.

You can choose:

* **Heuristics only**, or
* **Heuristics + LLM fallback**, or
* **LLM only** (but still with clear rules).

---

## 1. What each image model is actually for

### FLUX.1‑dev (`flux_dev`)

* 12B rectified flow transformer, open weights.([Hugging Face][1])
* Advertised strengths:

  * “Cutting‑edge output quality” and strong prompt following.
  * Very good at complex prompts and anatomically accurate images (hands, faces, etc.).
  * General‑purpose: portraits, environments, stylized art, etc.
* **Use it when**:

  * The prompt is a “normal” image: scenes, characters, stylized art, concept art.
  * There isn’t a clear bias towards “logo/text layout” or “super‑photoreal portrait”.

### RealVisXL V4.0 (`realvis_xl`)

* SDXL‑based fine‑tune “aimed at photorealism”.([Hugging Face][2])
* Primarily used for:

  * Faces, portraits, fashion, lifestyle photos.
  * Close‑up skin, eyes, emotions, etc.
* **Use it when**:

  * Prompt clearly describes a **realistic human** (or group) in photographic terms:

    * “portrait of a woman”, “studio photo”, “fashion shoot”, “RAW photo”, “35mm photo”, “headshot”, “selfie”, etc.
  * Style: “realistic”, “photorealistic”, “real life”, “RAW” rather than “anime / cartoon / flat illustration”.

### Stable Diffusion 3 Medium (`sd3_medium`)

* SD3 Medium is a ~2B parameter Multimodal Diffusion Transformer.([Hugging Face][3])
* Strengths:

  * Greatly improved **typography** (text in the image).
  * Better understanding of **complex prompts** (multi‑object, multi‑clause).
  * Efficient and scalable.
* **Use it when**:

  * Prompt looks like a **poster / UI / multi‑object layout**:

    * “movie poster”, “magazine cover”, “landing page UI”, “dashboard”, “infographic”, “comic page”, “four panels”.
  * There’s a lot of text to render **inside** the image (“title text”, “slogan”, “headline”, “big bold text”).
  * Prompt is long and compositional (many “and”, “with”, “in the background”, etc.).

### SDXL Base 1.0 (`logo_sdxl`)

* SDXL Base 1.0 is a 6.6B latent diffusion model for 1024×1024 images, with significantly better composition, faces, and **legible text** than SD 1.5/2.0.([Hugging Face][4])
* People often pair SDXL with logo/text LoRAs (e.g. logo‑focused fine‑tunes).([Reddit][5])
* **Use it when**:

  * Prompt explicitly mentions **logos / icons / badges / app icons / wordmarks**.
  * You want a **simple, flat, clean composition** where the *text itself* (brand name, short slogan) must be legible.

So qualitatively:

* **Portrait / realistic person → `realvis_xl` (RealVisXL)**
* **Logo / icon / strong text inside image → `logo_sdxl` (SDXL Base) or `sd3_medium`**
* **Super complex layout / multi‑object / UI / posters → `sd3_medium`**
* **Everything else → `flux_dev`**

---

## 2. Fast router architecture

Your “fast model” is a **small LLM** that just reads the prompt and emits a tiny JSON. Something like **Llama‑3‑8B‑Instruct** is ideal: it’s instruction‑tuned, strong at understanding text, and small enough to run locally easily on a single GPU.([Hugging Face][6])

On your 5090, a quantized 8B model will be *way* cheaper than running any diffusion model – we’re talking tens of milliseconds versus seconds for generation, so it’s totally reasonable as a “router”.

Pipeline per request:

1. User prompt arrives.
2. (Optional) **Cheap heuristic pre‑pass** (regex/length‑based) to get obvious cases.
3. Prepare a **router prompt** (system + few‑shot examples + the user prompt).
4. Call router LLM → it outputs a strict JSON like:

   ```json
   {
     "chosen_model_id": "realvis_xl",
     "scores": {
       "flux_dev": 0.30,
       "realvis_xl": 0.92,
       "sd3_medium": 0.25,
       "logo_sdxl": 0.05
     },
     "tags": ["portrait", "photorealistic"],
     "reason": "Prompt describes a realistic headshot of a person."
   }
   ```
5. Your backend parses JSON into `RoutingDecision` and picks `chosen_model_id`.

You can also *log* the JSON to `generation_logs.router_json` so you have a full audit trail and can fine‑tune later.

---

## 3. Concrete scoring logic (rule‑based) you can drop into Python

Even if you use an LLM, it helps to have **explicit code** both as fallback and as something you can show in the LLM’s system prompt as “rules”.

Here’s a concrete, fast function:

```python
import re
from collections import defaultdict
from typing import Dict, List, Tuple

ImageModelId = str  # "flux_dev", "realvis_xl", "sd3_medium", "logo_sdxl"

PORTRAIT_WORDS = [
    "portrait", "selfie", "headshot", "close-up", "close up", "bust shot",
    "studio photo", "studio portrait", "fashion shoot", "beauty shot",
    "model pose", "profile photo", "id photo", "passport photo",
    "face of", "photo of a man", "photo of a woman",
    "realistic man", "realistic woman", "realistic person"
]

HUMAN_WORDS = [
    "man", "woman", "boy", "girl", "person", "people", "child", "children",
    "teen", "adult", "old man", "old woman", "elderly"
]

PHOTO_WORDS = [
    "photo", "photograph", "photography", "dslr", "35mm", "canon", "nikon",
    "f/1.8", "f/2.8", "bokeh", "raw photo", "raw photograph",
    "high resolution photo", "8k photo", "real life", "realistic"
]

LOGO_WORDS = [
    "logo", "icon", "badge", "emblem", "crest", "monogram", "shield logo",
    "wordmark", "brand mark", "app icon", "favicon",
    "business card logo", "company logo", "branding"
]

TEXT_LAYOUT_WORDS = [
    "typography", "headline", "title text", "big title",
    "poster with text", "flyer", "brochure", "magazine cover",
    "book cover", "movie poster", "album cover", "banner",
    "billboard", "sign", "signage", "label", "package design",
    "infographic"
]

UI_LAYOUT_WORDS = [
    "ui", "user interface", "dashboard", "web app ui", "landing page",
    "mobile app screen", "wireframe", "mockup", "website design"
]

STYLE_CARTOON_WORDS = [
    "cartoon", "anime", "manga", "pixar style", "3d render",
    "illustration", "flat illustration", "vector art", "clip art"
]

LANDSCAPE_WORDS = [
    "landscape", "scenery", "mountain", "forest", "lake", "river",
    "ocean", "beach", "cityscape", "skyline", "sunset", "sunrise",
    "valley", "desert", "countryside", "fields"
]

def _contains_any(prompt: str, word_list: List[str]) -> bool:
    p = prompt.lower()
    return any(w in p for w in word_list)

def _count_multi_clause_markers(prompt: str) -> int:
    # crude measure of complexity
    p = prompt.lower()
    markers = [" and ", " with ", " while ", " in the background", " beside ", " behind "]
    return sum(p.count(m) for m in markers)

def score_models_for_prompt(prompt: str) -> Tuple[ImageModelId, Dict[ImageModelId, float], List[str], str]:
    """
    Core fast heuristic router.

    Returns:
        chosen_model_id, scores_dict, tags, reason
    """
    p = prompt.strip()
    p_lower = p.lower()
    tags: List[str] = []
    scores = defaultdict(float)

    # base priors (if nothing fires, flux_dev wins)
    scores["flux_dev"] = 0.6
    scores["realvis_xl"] = 0.2
    scores["sd3_medium"] = 0.1
    scores["logo_sdxl"] = 0.1

    # ---- 1. Detect portrait / human photo -----------------------------------
    has_portrait = _contains_any(p, PORTRAIT_WORDS)
    has_human = _contains_any(p, HUMAN_WORDS)
    has_photo_style = _contains_any(p, PHOTO_WORDS)

    if has_human:
        tags.append("human")
    if has_portrait:
        tags.append("portrait")
    if has_photo_style:
        tags.append("photo_style")

    if (has_human or has_portrait) and has_photo_style:
        # strong human photo case
        scores["realvis_xl"] += 0.6   # becomes dominant
        scores["flux_dev"] += 0.2
        scores["sd3_medium"] -= 0.1
        scores["logo_sdxl"] -= 0.2
    elif has_human or has_portrait:
        # human but style not clearly photographic
        scores["realvis_xl"] += 0.3
        scores["flux_dev"] += 0.3

    # ---- 2. Detect logos / text / branding ----------------------------------
    has_logo = _contains_any(p, LOGO_WORDS)
    has_text_layout = _contains_any(p, TEXT_LAYOUT_WORDS)
    has_ui_layout = _contains_any(p, UI_LAYOUT_WORDS)

    if has_logo:
        tags.append("logo")
    if has_text_layout:
        tags.append("text_layout")
    if has_ui_layout:
        tags.append("ui_layout")

    if has_logo:
        # strongly bias to logo_sdxl + sd3_medium
        scores["logo_sdxl"] += 0.7
        scores["sd3_medium"] += 0.3
        scores["realvis_xl"] -= 0.2
    if has_text_layout or has_ui_layout:
        scores["sd3_medium"] += 0.4
        scores["flux_dev"] += 0.1

    # ---- 3. Complexity / multi-object detection -----------------------------
    prompt_len = len(p.split())
    multi_clause = _count_multi_clause_markers(p)

    if prompt_len > 40 or multi_clause >= 3:
        tags.append("complex_scene")
        scores["sd3_medium"] += 0.4
        scores["flux_dev"] += 0.2

    # ---- 4. Stylized vs photoreal tweaks ------------------------------------
    has_cartoon_style = _contains_any(p, STYLE_CARTOON_WORDS)
    has_landscape = _contains_any(p, LANDSCAPE_WORDS)

    if has_cartoon_style:
        tags.append("stylized")
        scores["flux_dev"] += 0.4
        scores["realvis_xl"] -= 0.2

    if has_landscape:
        tags.append("landscape")
        scores["flux_dev"] += 0.2
        scores["sd3_medium"] += 0.1

    # ---- 5. Clamp and normalize ---------------------------------------------
    for k in list(scores.keys()):
        scores[k] = max(0.0, scores[k])  # no negatives
    total = sum(scores.values()) or 1.0
    for k in scores.keys():
        scores[k] /= total

    # ---- 6. Choose model with tie-break rules -------------------------------
    # order of preference in tie: realvis_xl, logo_sdxl, sd3_medium, flux_dev
    ordered_ids = ["realvis_xl", "logo_sdxl", "sd3_medium", "flux_dev"]
    best_id = None
    best_score = -1.0
    for mid in ordered_ids:
        s = scores[mid]
        if s > best_score:
            best_score = s
            best_id = mid

    # ---- 7. Reason string ----------------------------------------------------
    parts = []
    if "portrait" in tags or "human" in tags:
        parts.append("Prompt mentions humans/portraits, so RealVisXL and FLUX are favored.")
    if "logo" in tags or "text_layout" in tags or "ui_layout" in tags:
        parts.append("Prompt mentions logos or text/layout, so SD3-Medium and SDXL-logo are favored.")
    if "complex_scene" in tags:
        parts.append("Prompt is long/complex with many clauses, so SD3-Medium is boosted.")
    if "stylized" in tags:
        parts.append("Stylized/cartoon keywords bias towards FLUX.")
    if not parts:
        parts.append("No strong pattern detected; using FLUX as a general default.")

    reason = " ".join(parts)

    return best_id, dict(scores), tags, reason
```

This alone is a perfectly fine **router** – no LLM needed. If you’d like, you can:

* Use this **as your router**.
* Or, use this to generate a **“first guess”** and only call the LLM when scores are close (e.g. top two within 0.05).

---

## 4. How to plug a small LLM on top of that logic

Your “fast model” is a small LLM. Let’s assume **Meta‑Llama‑3‑8B‑Instruct**. It’s instruction‑tuned and strong at structured output and prompt understanding.([Hugging Face][6])

### 4.1. Router JSON schema (what LLM returns)

You already have something like:

```python
class RoutingDecision(BaseModel):
    chosen_model_id: Literal["flux_dev", "realvis_xl", "sd3_medium", "logo_sdxl"]
    scores: Dict[str, float]  # {"flux_dev": 0.7, ...}
    tags: List[str]
    reason: str
    # (plus prank-related fields if you want)
```

The LLM’s output should always match this schema.

### 4.2. System prompt with explicit model rules

This is where we bake the **detailed logic** in. Below is a system prompt you can literally drop into `build_router_prompt()` in `router.py`, adapted for Llama‑3 8B (but works for any decently aligned instruct model):

```text
You are an "image model router". Your only job is to read a user's text prompt
and decide which ONE of these image generation engines is best:

- flux_dev:
    A 12B parameter FLUX.1-dev rectified flow text-to-image model.
    Strengths: general-purpose, very strong prompt following, handles complex
    scenes, good anatomy and hands, works for both stylized and realistic images.

- realvis_xl:
    RealVisXL V4.0, an SDXL-based fine-tune aimed at photorealism and portraits.
    Strengths: highly realistic faces, skin, eyes, and people. Best when the
    prompt clearly describes a human person in a photographic style.

- sd3_medium:
    Stable Diffusion 3 Medium. 2B parameter MMDiT text-to-image model.
    Strengths: strong understanding of complex, multi-part prompts and
    better typography (text inside the image), suitable for posters, covers,
    UI mockups, dashboards, infographics, and scenes with many elements.

- logo_sdxl:
    SDXL-based model suitable for logos, icons, simple shapes and flat,
    text-centric designs. Best when user explicitly asks for a logo, icon,
    badge, or simple design with readable text.

You MUST analyze the prompt in terms of:
- "portrait / human photo"
- "logo / icon / branding"
- "text layout / poster / UI"
- "complex multi-object scene"
- "stylized cartoon / anime / illustration"
- "landscape / environment"

Use these rules:

1. HUMAN PORTRAITS:
   If the prompt clearly mentions a person or people (man, woman, boy, girl,
   person, people, child, etc) AND uses photographic language (photo, photograph,
   DSLR, 35mm, RAW, headshot, close-up, studio photo, fashion shoot, etc),
   then:
     - Give realvis_xl a score >= 0.85.
     - Give flux_dev a secondary score in [0.5, 0.7].
     - Keep sd3_medium <= 0.4 and logo_sdxl <= 0.3 unless the prompt ALSO
       clearly asks for a logo.

2. LOGOS / ICONS / BRANDING:
   If the prompt includes words like logo, icon, badge, emblem, crest, app icon,
   wordmark, brand mark, business card, or simple flat logo:
     - Give logo_sdxl a score >= 0.8.
     - Give sd3_medium a supporting score in [0.5, 0.7] if there is text layout.
     - Keep realvis_xl <= 0.3 unless there is also a portrait.
     - Keep flux_dev <= 0.5.

3. TEXT / POSTERS / UI LAYOUT:
   If the prompt describes posters, flyers, book covers, magazine covers, movie
   posters, banners, infographics, dashboards, or UI screens, OR explicitly
   asks for big title text, typography, or headlines:
     - Give sd3_medium a score >= 0.8.
     - Give flux_dev a moderate score in [0.4, 0.6].
     - logo_sdxl can also get a moderate score if it looks like a logo+text design.

4. COMPLEX SCENES:
   If the prompt is long (more than ~40 words) or has many clauses joined by
   "and", "with", "in the background", etc, treat it as a complex scene:
     - Increase sd3_medium and flux_dev scores modestly.
     - Only prefer realvis_xl if there is strong portrait/human evidence.
     - Only prefer logo_sdxl if there are clear logo/branding keywords.

5. STYLIZED / CARTOON:
   If the style is clearly cartoon, anime, manga, vector art, flat illustration,
   or 3D render (Pixar style, toon style, etc), and not strongly about realistic
   faces:
     - Give flux_dev the highest score (>= 0.8).
     - sd3_medium can get a moderate score if the prompt is complex.

6. LANDSCAPES / ENVIRONMENTS:
   If the prompt mostly talks about landscapes, scenery, nature, cityscapes,
   with no strong human portrait or logo/text emphasis:
     - Prefer flux_dev (>= 0.7) as a general strong model.
     - Give sd3_medium a secondary score in [0.4, 0.6].

7. TIE-BREAKING:
   - If human portrait indicators are strong, break ties in favor of realvis_xl.
   - If logo/branding indicators are strong, break ties in favor of logo_sdxl.
   - If complex + text/UI indicators are strong, break ties in favor of sd3_medium.
   - Otherwise, break ties in favor of flux_dev as a general model.

Your OUTPUT MUST BE A SINGLE JSON OBJECT, NO OTHER TEXT, with this exact shape:

{
  "chosen_model_id": "<one of: flux_dev, realvis_xl, sd3_medium, logo_sdxl>",
  "scores": {
    "flux_dev": 0.0,
    "realvis_xl": 0.0,
    "sd3_medium": 0.0,
    "logo_sdxl": 0.0
  },
  "tags": ["tag1", "tag2"],
  "reason": "1-3 sentences explaining the choice."
}

- The scores must be between 0.0 and 1.0 and should sum to approximately 1.0.
- The "tags" array should contain short keywords like "portrait", "logo",
  "text_layout", "complex_scene", "stylized", "landscape".
- The "chosen_model_id" must match the model with the highest score after your
  reasoning and tie-break rules.

Now analyze the user's prompt and respond ONLY with the JSON object, nothing else.
```

Then in code, you wrap this into something like:

```python
def build_router_prompt(user_prompt: str) -> str:
    system_and_instructions = <the big string above>
    full_prompt = f"{system_and_instructions}\n\nUSER PROMPT:\n{user_prompt.strip()}\n"
    return full_prompt
```

Call Llama‑3‑8B‑Instruct with that prompt, parse JSON, and you’re done.

**Why this is fast enough:** generating ~200–400 tokens of JSON + reasoning with an 8B model on a 5090 is tiny compared to running a 20–30 step diffusion pipeline at 1024×1024. And you can further quantize the LLM to 4‑bit without impacting reasoning quality much.

---

## 5. How to combine heuristics + LLM for even more speed

You can do:

1. **Run `score_models_for_prompt(prompt)`** (the Python rule‑based router).
2. If one model’s score > 0.7 and the gap to second best > 0.2:

   * Just use that model; **skip LLM** entirely.
3. Otherwise (ambiguous cases):

   * Call Llama‑3 8B with the system prompt above.
   * Use LLM’s JSON decision.

Pseudo‑router:

```python
def route_model_fast(prompt: str) -> RoutingDecision:
    heuristic_choice, heuristic_scores, tags, reason = score_models_for_prompt(prompt)

    # strong confidence threshold
    sorted_scores = sorted(heuristic_scores.items(), key=lambda kv: kv[1], reverse=True)
    best_id, best_score = sorted_scores[0]
    second_id, second_score = sorted_scores[1]

    if best_score >= 0.7 and (best_score - second_score) >= 0.2:
        # no need to call LLM; we are confident
        return RoutingDecision(
            chosen_model_id=best_id,
            scores=heuristic_scores,
            tags=tags,
            reason=f"Heuristic router: {reason}"
        )

    # otherwise call LLM router
    router = get_router()  # wraps Llama-3-8B
    decision = router.route(prompt)  # uses the JSON schema

    return decision
```

This gives you **three layers** of control:

* Hard‑coded behavior for the obvious cases (fast).
* LLM‑driven behavior for ambiguous prompts.
* Full JSON logs saved for training/fine‑tuning later.

---

## 6. Summary for your coding agent

If you want to hand this to the VS Code/Codex agent, you can say:

> Implement a model router that:
>
> 1. Uses the `score_models_for_prompt(prompt)` Python function (provided) to compute initial scores for four engines: `flux_dev`, `realvis_xl`, `sd3_medium`, `logo_sdxl`.
> 2. If the best model has score ≥ 0.7 and is at least 0.2 higher than the second best, use that engine directly.
> 3. Otherwise, call a small local LLM (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`) with the system prompt that encodes the detailed routing rules and expects JSON output with fields `chosen_model_id`, `scores`, `tags`, and `reason`.
> 4. Parse the JSON into the existing `RoutingDecision` Pydantic model and use `chosen_model_id` to select the image engine.
> 5. Log the entire `RoutingDecision` JSON into `generation_logs.router_json` for every request.

This way you have **explicit**, **fast**, and **LLM‑aware** logic for how the backend decides which image model to use for every prompt.

[1]: https://huggingface.co/black-forest-labs/FLUX.1-dev?utm_source=chatgpt.com "black-forest-labs/FLUX.1-dev"
[2]: https://huggingface.co/SG161222/RealVisXL_V4.0?utm_source=chatgpt.com "SG161222/RealVisXL_V4.0"
[3]: https://huggingface.co/stabilityai/stable-diffusion-3-medium?utm_source=chatgpt.com "stabilityai/stable-diffusion-3-medium"
[4]: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0?utm_source=chatgpt.com "stabilityai/stable-diffusion-xl-base-1.0"
[5]: https://www.reddit.com/r/StableDiffusion/comments/15kufbl/logoredmond_the_lora_for_creating_logos_in_sd_xl/?utm_source=chatgpt.com "Logo.Redmond. The Lora for creating LOGOS in SD XL 1.0 ..."
[6]: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct?utm_source=chatgpt.com "meta-llama/Meta-Llama-3-8B-Instruct"
