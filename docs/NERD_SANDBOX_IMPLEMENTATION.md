# Nerd Sandbox - Backend Implementation Summary

**Date:** November 26, 2025  
**Feature:** RTX 5090 Benchmark Testing System ("Nerd Sandbox")  
**Frontend Status:** ✅ Complete  
**Backend Status:** ⏳ Pending Implementation

---

## Overview

The **Nerd Sandbox** is a new page at `/nerd-sandbox` that allows users to benchmark three AI models (Flux.1-dev, RealVis XL V4.0, SD3 Medium) on the RTX 5090 GPU with custom parameters. Users can:

1. Select a prompt (or randomize from 100 curated test prompts)
2. Configure global settings (resolution: 512×512 or 1024×1024, TF32 enabled/disabled)
3. Adjust per-model parameters (steps and guidance scale with model-specific ranges)
4. Run a sequential benchmark test across all three models
5. View real-time nerdy status updates (queued → loading model → processing → finished)
6. See results with images, timing data, and parameters

This feature showcases the GPU capabilities and provides transparency for technical users who want to understand model performance characteristics.

---

## Frontend Implementation (Complete)

### Files Created/Modified:

1. **`public/data/benchmark_prompts_v2.json`** - 100 curated test prompts across 5 categories
2. **`client/src/lib/api.ts`** - Benchmark types and API functions added
3. **`client/src/pages/NerdSandbox.tsx`** - Full page component (650+ lines)
4. **`client/src/App.tsx`** - Route registered at `/nerd-sandbox`
5. **`client/src/components/Layout.tsx`** - Navigation links added (desktop + mobile)
6. **`client/public/sitemap.xml`** - SEO entry added

### UI Features:

- **Prompt Selector**: 3-line textarea + randomize button 🎲
- **Global Controls**: Resolution radio (512 default) + TF32 checkbox (enabled default)
- **Model Sliders**: Per-model steps and guidance sliders with tooltips
  - Flux: 2-5 steps, 0.0-2.0 guidance
  - RealVis: 10-48 steps, 1.0-8.0 guidance
  - SD3: 18-48 steps, 3.0-9.0 guidance
- **Run Button**: Starts benchmark, disabled during run
- **Animated Status**: Smooth transitions showing queue → model loading → processing → finished (1s) → next model
- **Results Display**: Vertical cards showing images, timing, and parameters for each model

### Polling Architecture:

- Frontend polls `GET /api/bench/5090/{run_id}` every **1.5 seconds** during active run
- Stops polling when status is "done" or "error"
- Displays 1-second "finished" animation when each model completes before moving to next

---

## Backend API Requirements

### Endpoint 1: Enqueue Benchmark

**POST** `/api/bench/5090`

#### Request Body:
```json
{
  "prompt": "a corgi astronaut floating in space",
  "resolution": 512,
  "tf32_enabled": true,
  "engines": [
    { "engine": "flux_dev", "steps": 4, "guidance": 1.0 },
    { "engine": "realvis_xl", "steps": 28, "guidance": 4.5 },
    { "engine": "sd3_medium", "steps": 30, "guidance": 5.0 }
  ]
}
```

#### Response:
```json
{
  "run_id": "bench_abc123xyz"
}
```

**Status Code:** 200 OK  
**Error Handling:** Return 400 if validation fails, 500 if queue is full

---

### Endpoint 2: Poll Benchmark Status

**GET** `/api/bench/5090/{run_id}`

#### Response (Queued):
```json
{
  "run_id": "bench_abc123xyz",
  "prompt": "a corgi astronaut floating in space",
  "status": "queued",
  "current_engine": null,
  "resolution": 512,
  "tf32_enabled": true,
  "engines": {
    "flux_dev": { "status": "pending", "steps": 4, "guidance": 1.0 },
    "realvis_xl": { "status": "pending", "steps": 28, "guidance": 4.5 },
    "sd3_medium": { "status": "pending", "steps": 30, "guidance": 5.0 }
  }
}
```

#### Response (Running - Model Loading):
```json
{
  "run_id": "bench_abc123xyz",
  "prompt": "a corgi astronaut floating in space",
  "status": "running",
  "current_engine": "flux_dev",
  "resolution": 512,
  "tf32_enabled": true,
  "engines": {
    "flux_dev": { "status": "pending", "steps": 4, "guidance": 1.0 },
    "realvis_xl": { "status": "pending", "steps": 28, "guidance": 4.5 },
    "sd3_medium": { "status": "pending", "steps": 30, "guidance": 5.0 }
  }
}
```

#### Response (Running - Processing):
```json
{
  "run_id": "bench_abc123xyz",
  "prompt": "a corgi astronaut floating in space",
  "status": "running",
  "current_engine": "flux_dev",
  "resolution": 512,
  "tf32_enabled": true,
  "engines": {
    "flux_dev": { "status": "running", "steps": 4, "guidance": 1.0 },
    "realvis_xl": { "status": "pending", "steps": 28, "guidance": 4.5 },
    "sd3_medium": { "status": "pending", "steps": 30, "guidance": 5.0 }
  }
}
```

#### Response (Running - First Model Done):
```json
{
  "run_id": "bench_abc123xyz",
  "prompt": "a corgi astronaut floating in space",
  "status": "running",
  "current_engine": "realvis_xl",
  "resolution": 512,
  "tf32_enabled": true,
  "engines": {
    "flux_dev": {
      "status": "done",
      "elapsed_ms": 3247,
      "image_url": "https://app.promptpics.ai/static/bench/bench_abc123xyz_flux_dev.png",
      "steps": 4,
      "guidance": 1.0
    },
    "realvis_xl": { "status": "pending", "steps": 28, "guidance": 4.5 },
    "sd3_medium": { "status": "pending", "steps": 30, "guidance": 5.0 }
  }
}
```

#### Response (Complete):
```json
{
  "run_id": "bench_abc123xyz",
  "prompt": "a corgi astronaut floating in space",
  "status": "done",
  "current_engine": null,
  "resolution": 512,
  "tf32_enabled": true,
  "engines": {
    "flux_dev": {
      "status": "done",
      "elapsed_ms": 3247,
      "image_url": "https://app.promptpics.ai/static/bench/bench_abc123xyz_flux_dev.png",
      "steps": 4,
      "guidance": 1.0
    },
    "realvis_xl": {
      "status": "done",
      "elapsed_ms": 5821,
      "image_url": "https://app.promptpics.ai/static/bench/bench_abc123xyz_realvis_xl.png",
      "steps": 28,
      "guidance": 4.5
    },
    "sd3_medium": {
      "status": "done",
      "elapsed_ms": 6103,
      "image_url": "https://app.promptpics.ai/static/bench/bench_abc123xyz_sd3_medium.png",
      "steps": 30,
      "guidance": 5.0
    }
  }
}
```

#### Response (Error):
```json
{
  "run_id": "bench_abc123xyz",
  "prompt": "a corgi astronaut floating in space",
  "status": "error",
  "current_engine": "realvis_xl",
  "resolution": 512,
  "tf32_enabled": true,
  "engines": {
    "flux_dev": {
      "status": "done",
      "elapsed_ms": 3247,
      "image_url": "https://app.promptpics.ai/static/bench/bench_abc123xyz_flux_dev.png",
      "steps": 4,
      "guidance": 1.0
    },
    "realvis_xl": { "status": "error", "steps": 28, "guidance": 4.5 },
    "sd3_medium": { "status": "pending", "steps": 30, "guidance": 5.0 }
  }
}
```

**Status Code:** 200 OK  
**Error Handling:** Return 404 if run_id not found

---

## Backend Processing Flow

### Execution Order (Sequential):

1. **Enqueue Request**
   - Generate unique `run_id` (e.g., `bench_{timestamp}_{random}`)
   - Validate parameters (prompt, resolution, engines array)
   - Add to benchmark queue
   - Return `run_id` immediately

2. **Queue Processing**
   - Status: `"queued"`, `current_engine: null`, all engines `"pending"`

3. **Model 1: Flux.1-dev**
   - Update: `status: "running"`, `current_engine: "flux_dev"`
   - Update: Flux status `"pending"` (model loading)
   - Update: Flux status `"running"` (processing)
   - Generate image with specified steps/guidance at resolution
   - Save image to static path: `/static/bench/{run_id}_flux_dev.png`
   - Update: Flux status `"done"`, add `elapsed_ms` and `image_url`

4. **Model 2: RealVis XL V4.0**
   - Update: `current_engine: "realvis_xl"`
   - Update: RealVis status `"pending"` (model loading)
   - Update: RealVis status `"running"` (processing)
   - Generate image with specified steps/guidance at resolution
   - Save image to static path: `/static/bench/{run_id}_realvis_xl.png`
   - Update: RealVis status `"done"`, add `elapsed_ms` and `image_url`

5. **Model 3: SD3 Medium**
   - Update: `current_engine: "sd3_medium"`
   - Update: SD3 status `"pending"` (model loading)
   - Update: SD3 status `"running"` (processing)
   - Generate image with specified steps/guidance at resolution
   - Save image to static path: `/static/bench/{run_id}_sd3_medium.png`
   - Update: SD3 status `"done"`, add `elapsed_ms` and `image_url`

6. **Complete**
   - Update: `status: "done"`, `current_engine: null`

### Storage:

- Store run state in memory (Redis recommended for multi-worker setups)
- TTL: 1 hour (runs expire after completion)
- Images saved to disk at `/static/bench/` (auto-cleanup after 24 hours recommended)

### Error Handling:

- If any model fails, set that engine's status to `"error"`
- Continue to next model (don't abort entire run)
- Set overall status to `"error"` if any model fails
- Log errors but return graceful error state to frontend

---

## Frontend Status Display Logic

The frontend displays status based on the following priority:

1. **"Finished" animation** (1 second): When engine transitions from `"running"` → `"done"`
   - Shows: `✅ Finished {Model Name}`
   - Green pulsing animation

2. **"Processing"**: When `current_engine` is set AND that engine's status is `"running"`
   - Shows: `⚡ Processing {Model Name}...`
   - Indigo pulsing animation

3. **"Loading Model"**: When `current_engine` is set AND that engine's status is `"pending"`
   - Shows: `🔧 Loading {Model Name} model...`
   - Purple spinning animation

4. **"Queued"**: When overall status is `"queued"`
   - Shows: `Queued...`
   - Yellow pulsing animation

5. **"Complete"**: When overall status is `"done"`
   - Shows: `🎉 Benchmark Complete!`
   - Green checkmark

6. **"Error"**: When overall status is `"error"`
   - Shows: `❌ Benchmark Error`
   - Red error icon

---

## Parameter Validation

### Required Validation:

- **prompt**: Non-empty string, max 500 chars
- **resolution**: Must be `512` or `1024`
- **tf32_enabled**: Boolean
- **engines**: Array of 3 objects, each with:
  - `engine`: Must be one of `["flux_dev", "realvis_xl", "sd3_medium"]`
  - `steps`: Integer within model's valid range
  - `guidance`: Float within model's valid range

### Model-Specific Ranges (Backend Enforcement):

```python
STEP_RANGES = {
    "flux_dev": (2, 5),
    "realvis_xl": (10, 48),
    "sd3_medium": (18, 48)
}

GUIDANCE_RANGES = {
    "flux_dev": (0.0, 2.0),
    "realvis_xl": (1.0, 8.0),
    "sd3_medium": (3.0, 9.0)
}
```

---

## Testing Scenarios

### Manual Testing:

1. **Basic Flow**:
   - Go to `/nerd-sandbox`
   - Click randomize 🎲 button → verify prompt changes
   - Adjust sliders → verify values update
   - Click "Run Test on 5090" → verify button disables
   - Watch status updates → verify smooth transitions
   - Wait for completion → verify all 3 results display

2. **Edge Cases**:
   - Empty prompt → verify error message
   - Change resolution mid-run → verify no effect on current run
   - Rapid re-runs → verify queue handling
   - Navigate away during run → verify poll cleanup

3. **Error Handling**:
   - Backend returns 500 → verify error alert
   - Backend returns 404 on poll → verify error state
   - Backend returns malformed JSON → verify graceful degradation

### Backend Testing (Recommended):

```bash
# Test enqueue
curl -X POST https://app.promptpics.ai/api/bench/5090 \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "test prompt",
    "resolution": 512,
    "tf32_enabled": true,
    "engines": [
      {"engine": "flux_dev", "steps": 4, "guidance": 1.0},
      {"engine": "realvis_xl", "steps": 28, "guidance": 4.5},
      {"engine": "sd3_medium", "steps": 30, "guidance": 5.0}
    ]
  }'

# Test poll (replace run_id)
curl https://app.promptpics.ai/api/bench/5090/bench_abc123xyz
```

---

## Production Considerations

### Performance:

- Benchmarks run sequentially (3 models × ~5-10s each = ~15-30s total)
- Queue to prevent GPU overload
- Consider rate limiting per IP (e.g., 1 run per 5 minutes)

### Storage:

- Images saved to `/static/bench/` directory
- Implement cleanup job (delete files >24 hours old)
- Consider S3/CloudFlare R2 for production scale

### Monitoring:

- Log all benchmark requests (prompt, params, user IP)
- Track completion times for performance monitoring
- Alert on high error rates or queue backlog

### Security:

- No authentication required (public feature)
- Rate limiting recommended
- Content filtering on prompts (NSFW/illegal content)
- File size limits on generated images

---

## Frontend Code References

- **Types**: `client/src/lib/api.ts` (lines 522-592)
- **Page Component**: `client/src/pages/NerdSandbox.tsx`
- **Prompts Data**: `public/data/benchmark_prompts_v2.json`
- **Route**: `client/src/App.tsx` (line 58)
- **Navigation**: `client/src/components/Layout.tsx` (lines 101, 135)

---

## Questions for Backend Team

1. **Queue Architecture**: Should we use Redis for state storage or in-memory?
2. **Concurrency**: How many concurrent benchmarks can the 5090 handle?
3. **Rate Limiting**: Should we implement IP-based rate limiting?
4. **Image Storage**: Local disk or cloud storage (S3/R2)?
5. **Cleanup Strategy**: Automated cleanup job or manual purge?
6. **Error Logging**: Integration with existing logging infrastructure?
7. **Analytics**: Should we track benchmark usage for performance insights?

---

## Timeline

- **Frontend**: ✅ Complete (November 26, 2025)
- **Backend**: ⏳ Awaiting implementation
- **Testing**: After backend deployment
- **Launch**: After successful end-to-end testing

---

**Contact**: Frontend implementation complete and ready for backend integration. Frontend will poll every 1.5 seconds and expects the exact response formats documented above.
