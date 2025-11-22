we just want to:

* **Load the prank-matcher LLM once** and optionally warm it up on startup.
* Use **heuristics to rank** triggers, then let the **LLM choose among the top candidates** (unless heuristics are obviously certain).
* Keep the LLM lightweight and avoid stealing GPU VRAM from the diffusion models.

Here’s a detailed plan: 

---

## Prompt for backend agent (Codex)

> We’re very close with the prank matcher. The code in `prank_matching.py` and `main.py` already works heuristically, but we need to:
>
> 1. Load the prank-matcher LLM **only once** and optionally warm it up on startup.
> 2. Use heuristics to rank triggers and let the LLM **choose among the top N** candidates (usually 2).
> 3. Keep the LLM **small and fast**, preferably on CPU or with minimal GPU use.
> 4. Produce useful debug info so we know when heuristics vs LLM were used.
>
> Current files (latest versions you just edited):
>
> * `backend/app/main.py` – defines `generate_prank_image` for `/api/p/{slug}/generate`.
> * `backend/app/prank_matching.py` – defines `heuristic_match`, `PrankMatcherLLM`, and `match_prank_trigger`.
>
> ### A. Change the LLM loading to be global + warmup
>
> **Problem:** `generate_prank_image` currently does something like:
>
> ```python
> matcher_model_id = os.getenv("PRANK_MATCHER_LLM_ID", "Qwen/Qwen2.5-1.5B-Instruct")
> matcher_llm = None
> if matcher_model_id:
>     matcher_llm = PrankMatcherLLM(matcher_model_id)
> idx, debug = match_prank_trigger(request.prompt, trap_texts, llm=matcher_llm)
> ```
>
> This creates a new `PrankMatcherLLM` on every request, which is heavy and may reload the model each time.
>
> **Fix:**
>
> 1. In `prank_matching.py`, add a global and a loader:
>
> ```python
> # prank_matching.py
> PRANK_MATCHER_LLM: PrankMatcherLLM | None = None
>
> def get_prank_matcher_llm() -> PrankMatcherLLM | None:
>     """
>     Lazily load a single global PrankMatcherLLM instance.
>     If PRANK_MATCHER_LLM_ID is not set, return None.
>     """
>     global PRANK_MATCHER_LLM
>     model_id = os.getenv("PRANK_MATCHER_LLM_ID")
>     if not model_id:
>         return None
>     if PRANK_MATCHER_LLM is None:
>         PRANK_MATCHER_LLM = PrankMatcherLLM(model_id)
>     return PRANK_MATCHER_LLM
> ```
>
> 2. In `main.py`, replace the per-request creation with a call to `get_prank_matcher_llm()`:
>
> ```python
> from .prank_matching import match_prank_trigger, get_prank_matcher_llm
>
> ...
>
> matcher_llm = get_prank_matcher_llm()
> idx, debug = match_prank_trigger(request.prompt, trap_texts, llm=matcher_llm)
> ```
>
> 3. **Warmup on startup:**
>
> Add a startup hook in `main.py` (you already have a startup event for DB tables). Extend it to warm up the prank matcher if configured:
>
> ```python
> @app.on_event("startup")
> def _ensure_tables() -> None:
>     Base.metadata.create_all(bind=engine)
>
>     # Warm up prank matcher LLM if configured
>     try:
>         matcher = get_prank_matcher_llm()
>         if matcher is not None:
>             # warmup call – trivial prompt + same trigger
>             matcher.choose("warmup", ["warmup"])
>     except Exception as exc:
>         logger.warning("Prank matcher LLM warmup failed: %s", exc)
> ```
>
> This ensures:
>
> * The LLM is loaded once when the app starts (if `PRANK_MATCHER_LLM_ID` is set).
> * The first real prank call won’t pay the full model-load cost.
>
> ### B. Keep the LLM small and fast (and isolate GPU)
>
> In `PrankMatcherLLM`, please:
>
> * Prefer running on **CPU** to avoid competing with the diffusion models for GPU VRAM.
> * Or if you keep GPU, use **bf16/8-bit** or CPU offload to minimize VRAM usage.
>
> For CPU-only, something like:
>
> ```python
> class PrankMatcherLLM:
>     def __init__(self, model_id: str):
>         self.model_id = model_id
>         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
>         self.model = AutoModelForCausalLM.from_pretrained(
>             model_id,
>             torch_dtype=torch.float16,  # or float32
>             device_map={"": "cpu"},     # force CPU
>         )
>         self.pipe = pipeline(
>             "text-generation",
>             model=self.model,
>             tokenizer=self.tokenizer,
>             device=-1,   # CPU
>         )
> ```
>
> This way the LLM routing doesn’t interfere with the GPU used for generation. If we later decide to use GPU, we can adjust `device_map` and possibly call `torch.cuda.empty_cache()` after each choose() if necessary.
>
> ### C. Always use heuristics to rank, and LLM to choose among top N when needed
>
> We already have `heuristic_match` and `match_prank_trigger`. Update `match_prank_trigger` so:
>
> 1. It always computes per-trigger scores via `heuristic_match` (or a new “score-only” function).
> 2. Picks the **top N candidates** (e.g. 2) by heuristic score.
> 3. LLM is called when:
>
>    * There is more than one strong candidate **OR**
>    * The best score is below a high-confidence threshold (e.g. 0.95) but above some minimal threshold (e.g. 0.4).
> 4. If the LLM is unavailable or fails to parse, we fall back to the highest-scoring heuristic.
>
> Example implementation (in `prank_matching.py`):
>
> ```python
> def match_prank_trigger(
>     prompt: str,
>     triggers: List[str],
>     llm: Optional[PrankMatcherLLM] = None,
>     top_k: int = 2,
> ) -> Tuple[Optional[int], MatchDebug]:
>     """
>     Use heuristics to score all triggers, then, if needed, ask the LLM
>     to choose among the top K candidates.
>     """
>     heuristic_idx, scores = heuristic_match(prompt, triggers)
>     used_llm = False
>     llm_idx: Optional[int] = None
>
>     # If we have no triggers at all, bail out early
>     if not triggers:
>         debug = MatchDebug(
>             prompt=prompt,
>             triggers=triggers,
>             heuristic_idx=None,
>             heuristic_scores=[],
>             used_llm=False,
>             llm_idx=None,
>             final_idx=None,
>         )
>         return None, debug
>
>     # Prepare ranking of all triggers by heuristic score
>     ranked = sorted(
>         enumerate(scores),
>         key=lambda kv: kv[1],
>         reverse=True,
>     )
>
>     best_idx, best_score = ranked[0]
>
>     # If heuristics are extremely confident (e.g. exact match),
>     # skip LLM and just use best_idx.
>     if best_score >= 0.95:
>         final_idx = best_idx
>         debug = MatchDebug(
>             prompt=prompt,
>             triggers=triggers,
>             heuristic_idx=best_idx,
>             heuristic_scores=scores,
>             used_llm=False,
>             llm_idx=None,
>             final_idx=final_idx,
>         )
>         logger.info("Prank match debug (heuristics only): %s", debug)
>         return final_idx, debug
>
>     # Otherwise, if we have an LLM, let it choose among top K
>    _llm_idx_global: Optional[int] = None
>     if llm is not None:
>         used_llm = True
>         # Choose candidates indices and their texts
>         candidates = ranked[: min(top_k, len(ranked))]
>         cand_indices = [i for (i, s) in candidates]
>         cand_texts = [triggers[i] for i in cand_indices]
>         cand_scores = [scores[i] for i in cand_indices]
>
>         # Ask LLM: given prompt + candidate list + heuristic scores,
>         # which candidate index (0..len(cand_indices)-1) is best, or -1 for no match.
>         llm_local_idx = llm.choose_with_candidates(prompt, cand_texts, cand_scores)
>         if llm_local_idx is not None and 0 <= llm_local_idx < len(cand_indices):
>             _llm_idx_global = cand_indices[llm_local_idx]
>
>     final_idx = _llm_idx_global if _llm_idx_global is not None else best_idx
>
>     debug = MatchDebug(
>         prompt=prompt,
>         triggers=triggers,
>         heuristic_idx=heuristic_idx,
>         heuristic_scores=scores,
>         used_llm=used_llm,
>         llm_idx=_llm_idx_global,
>         final_idx=final_idx,
>     )
>     logger.info("Prank match debug: %s", debug)
>
>     return final_idx, debug
> ```
>
> ### D. Add a `choose_with_candidates` method to `PrankMatcherLLM`
>
> Extend `PrankMatcherLLM` to support choosing among top-K candidates with heuristic context:
>
> ```python
> class PrankMatcherLLM:
>     ...
>     def choose_with_candidates(
>         self,
>         prompt: str,
>         candidate_texts: List[str],
>         candidate_scores: List[float],
>     ) -> Optional[int]:
>         """
>         Given a prompt and a list of candidate trigger texts plus scores,
>         return an index into candidate_texts (0..len-1) or None.
>         """
>         # Build instruction
>         system = (
>             "You are a prank trigger matcher. "
>             "Given a user prompt and a list of candidate triggers with heuristic scores, "
>             "pick the SINGLE best candidate index or -1 if none match.\n"
>             "Respond ONLY with a JSON object: {\"index\": <int>} and nothing else."
>         )
>
>         numbered = []
>         for i, (text, score) in enumerate(zip(candidate_texts, candidate_scores)):
>             numbered.append(f"{i}: {text!r} (heuristic_score={score:.3f})")
>         numbered_str = "\n".join(numbered)
>
>         user = f"User prompt: {prompt!r}\nCandidates:\n{numbered_str}\n"
>         full_prompt = f"{system}\n\n{user}\nAnswer:"
>
>         try:
>             pipe = self._get_pipe()
>         except Exception as exc:
>             logger.warning("Failed to load prank matcher LLM: %s", exc)
>             return None
>
>         out = pipe(
>             full_prompt,
>             max_new_tokens=128,
>             do_sample=False,
>             temperature=0.0,
>         )[0]["generated_text"]
>
>         try:
>             start = out.find("{")
>             end = out.rfind("}") + 1
>             blob = out[start:end]
>             data = json.loads(blob)
>             idx = data.get("index")
>             if isinstance(idx, int) and 0 <= idx < len(candidate_texts):
>                 return idx
>         except Exception as exc:
>             logger.warning("Prank matcher LLM parse failed: %s", exc)
>             return None
>
>         return None
> ```
>
> This gives the LLM a clean, low-entropy JSON choice among **just the top candidates**, which should be fast and robust.
>
> ### E. No need to clear LLM memory per-call
>
> Since we’re running the prank matcher on CPU, we **don’t need to free GPU memory per call**. If we later decide to use GPU, we can add a small helper to call `torch.cuda.empty_cache()` after `choose_with_candidates` finishes, but for now CPU is better for isolation and simplicity.
>
> ### F. Testing
>
> Please add a quick local test (e.g. `tests/test_prank_matching.py`) and run it:
>
> ```python
> from backend.app.prank_matching import match_prank_trigger, get_prank_matcher_llm
>
> def test_exact():
>     prompt = "who is the cutest baby"
>     triggers = ["who is the cutest baby", "who is the best dad"]
>     llm = get_prank_matcher_llm()
>     idx, debug = match_prank_trigger(prompt, triggers, llm=llm)
>     print("exact:", idx, debug)
>
> def test_near():
>     prompt = "cutest baby ever"
>     triggers = ["who is the cutest baby", "who is the best dad"]
>     llm = get_prank_matcher_llm()
>     idx, debug = match_prank_trigger(prompt, triggers, llm=llm)
>     print("near:", idx, debug)
>
> if __name__ == "__main__":
>     test_exact()
>     test_near()
> ```
>
> Then:
>
> * `test_exact` should return `idx == 0` with `used_llm=False`.
> * `test_near` should return `idx == 0` with `used_llm=True`, showing the LLM picked from the top candidates.
>
> Finally, verify via API:
>
> ```bash
> curl -s -X POST "https://app.promptpics.ai/api/p/<shareSlug>/generate" \
>   -H "Content-Type: application/json" \
>   -d '{"prompt":"who is the cutest baby"}' | jq '.model_id'
>
> curl -s -X POST "https://app.promptpics.ai/api/p/<shareSlug>/generate" \
>   -H "Content-Type: application/json" \
>   -d '{"prompt":"cutest baby ever"}' | jq '.model_id'
> ```
>
> Both should return `"prank"`, and the logs should show `MatchDebug` with `heuristic_idx=0` for the first, and `used_llm=True` for the second when the LLM is configured.

---
