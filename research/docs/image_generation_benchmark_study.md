# Image Generation Model Benchmark Study
## Hardware-Accelerated Performance Analysis on NVIDIA RTX 5090

**Author**: Stephen  
**Date**: November 22, 2025  
**Hardware**: NVIDIA RTX 5090, AMD Threadripper 9970X (32 cores), 128GB DDR5

---

## Executive Summary

This study presents a comprehensive benchmark analysis of six state-of-the-art image generation models, evaluating performance across multiple dimensions including generation speed, resolution scaling, parameter sensitivity, and content-specific capabilities. Using an NVIDIA RTX 5090 with TF32 acceleration, we systematically analyze model behavior across 50 diverse prompts and multiple parameter configurations.

---

## 1. Model Selection

### Current Models (4)
1. **FLUX.1-dev** (black-forest-labs/FLUX.1-dev)
   - Architecture: Flow-based diffusion, 12B parameters
   - Strengths: General purpose, excellent prompt adherence
   - Use Case: Baseline high-quality generation

2. **RealVisXL V4.0** (SG161222/Realistic_Vision_V5.1_noVAE)
   - Architecture: SDXL-based fine-tune
   - Strengths: Photorealistic humans, portraits
   - Use Case: Human/portrait benchmark

3. **Stable Diffusion 3 Medium** (stabilityai/stable-diffusion-3-medium)
   - Architecture: MMDiT (Multimodal Diffusion Transformer)
   - Strengths: Complex compositions, text rendering
   - Use Case: Text and composition complexity

4. **HiDream I1** (HiDream.ai/HiDream-I1-Full)
   - Architecture: 17B Sparse DiT with MoE
   - Strengths: Superior text/logo rendering
   - Use Case: Typography benchmark

### Additional Models to Add (2-3)

5. **SDXL Lightning** (ByteDance/SDXL-Lightning)
   - Architecture: Distilled SDXL, 2-4 step generation
   - Strengths: Ultra-fast inference (<1 second)
   - Use Case: Speed benchmark baseline
   - Why: Tests the speed vs quality tradeoff

6. **Playground v2.5** (playgroundai/playground-v2.5-1024px-aesthetic)
   - Architecture: SDXL fine-tune with aesthetic training
   - Strengths: Consistent aesthetic style (Midjourney-like)
   - Use Case: Aesthetic consistency benchmark
   - Why: Different training philosophy (aesthetic-focused)

---

## 2. Benchmark Parameters

### Fixed Parameters
- **TF32**: Enabled for all models (consistent precision)
- **Seed**: 42 (for reproducibility in quality tests)
- **Batch Size**: 1 (to measure pure generation time)
- **VAE**: Default for each model
- **Scheduler**: Default for each model (document which one)

### Variable Parameters

#### A. Resolution Scaling Test
```python
resolutions = [
    (256, 256),   # Low-res baseline
    (512, 512),   # SD 1.5 native
    (768, 768),   # Your current cap
    (1024, 1024), # SDXL native
    (1536, 1536), # High-res test (if VRAM allows)
]
```

#### B. Steps Analysis
```python
steps_configs = {
    "ultra_fast": 4,
    "fast": 12,
    "balanced": 24,
    "quality": 40,
    "maximum": 60
}
```

#### C. Guidance Scale Analysis
```python
guidance_scales = [1.0, 3.5, 5.0, 7.5, 10.0]
```

---

## 3. Prompt Dataset (50 Prompts)

### Distribution
- 20% Text/Typography (10 prompts)
- 30% People/Portraits (15 prompts)
- 20% Animals/Nature (10 prompts)
- 20% Landscapes/Architecture (10 prompts)
- 10% Abstract/Artistic (5 prompts)

### Prompt Complexity Levels
- Simple (1-5 words): 10 prompts
- Medium (10-20 words): 25 prompts
- Complex (30+ words): 15 prompts

```json
{
  "prompts": [
    {
      "id": 1,
      "category": "text",
      "complexity": "medium",
      "prompt": "A vintage coffee shop menu board with chalk lettering saying 'FRESH BREW $3.50' in elegant script",
      "evaluation_criteria": ["text_accuracy", "style_consistency"]
    },
    {
      "id": 2,
      "category": "text",
      "complexity": "simple",
      "prompt": "Neon sign: OPEN 24/7",
      "evaluation_criteria": ["text_clarity", "glow_effect"]
    },
    {
      "id": 3,
      "category": "text",
      "complexity": "complex",
      "prompt": "A weathered wooden sign at a crossroads with three arrows pointing different directions labeled 'Yesterday 5 miles', 'Tomorrow 10 miles', and 'Today Right Here'",
      "evaluation_criteria": ["text_accuracy", "composition", "metaphorical_interpretation"]
    },
    {
      "id": 4,
      "category": "people",
      "complexity": "medium",
      "prompt": "Professional headshot of a female CEO in her 40s, confident smile, modern office background",
      "evaluation_criteria": ["facial_features", "age_accuracy", "professional_setting"]
    },
    {
      "id": 5,
      "category": "people",
      "complexity": "complex",
      "prompt": "Three generations of women baking together in a sunlit kitchen, grandmother teaching granddaughter to knead dough while mother watches fondly",
      "evaluation_criteria": ["multiple_people", "age_diversity", "hand_accuracy", "interaction"]
    },
    {
      "id": 6,
      "category": "people",
      "complexity": "simple",
      "prompt": "Elderly man reading newspaper",
      "evaluation_criteria": ["age_representation", "hand_detail"]
    },
    {
      "id": 7,
      "category": "animals",
      "complexity": "medium",
      "prompt": "Golden retriever puppy playing with a red ball in autumn leaves",
      "evaluation_criteria": ["fur_texture", "motion", "seasonal_context"]
    },
    {
      "id": 8,
      "category": "animals",
      "complexity": "complex",
      "prompt": "A majestic owl perched on a gnarled branch under moonlight, its amber eyes reflecting stars, surrounded by fireflies",
      "evaluation_criteria": ["feather_detail", "lighting", "atmosphere"]
    },
    {
      "id": 9,
      "category": "landscape",
      "complexity": "medium",
      "prompt": "Misty mountain valley at sunrise with a winding river",
      "evaluation_criteria": ["atmospheric_perspective", "lighting", "natural_elements"]
    },
    {
      "id": 10,
      "category": "landscape",
      "complexity": "complex",
      "prompt": "Cyberpunk cityscape with neon reflections on wet streets, flying cars leaving light trails, holographic advertisements on skyscrapers",
      "evaluation_criteria": ["architectural_detail", "lighting_effects", "sci-fi_elements"]
    },
    {
      "id": 11,
      "category": "abstract",
      "complexity": "medium",
      "prompt": "Visualization of time as flowing liquid crystal",
      "evaluation_criteria": ["abstract_interpretation", "visual_coherence"]
    },
    {
      "id": 12,
      "category": "text",
      "complexity": "medium",
      "prompt": "Birthday cake with 'Happy 30th Birthday Sarah!' written in pink frosting",
      "evaluation_criteria": ["text_accuracy", "cake_realism"]
    },
    {
      "id": 13,
      "category": "people",
      "complexity": "medium",
      "prompt": "Basketball player mid-dunk, intense expression, stadium lights",
      "evaluation_criteria": ["motion_blur", "facial_expression", "athletic_pose"]
    },
    {
      "id": 14,
      "category": "animals",
      "complexity": "simple",
      "prompt": "Sleeping cat on windowsill",
      "evaluation_criteria": ["fur_detail", "peaceful_mood"]
    },
    {
      "id": 15,
      "category": "landscape",
      "complexity": "simple",
      "prompt": "Desert sunset",
      "evaluation_criteria": ["color_gradient", "simplicity"]
    }
  ]
}
```

---

## 4. Metrics to Capture

### Performance Metrics
1. **Generation Time** (per resolution/steps/guidance)
   - Cold start time (first generation)
   - Warm generation time (subsequent)
   - Time per step calculation
   - Memory usage (VRAM peak)

2. **Throughput Analysis**
   - Images per minute at different quality settings
   - Batch processing efficiency (if testing batch>1)

3. **Quality Metrics** (automated where possible)
   - CLIP score (prompt adherence)
   - FID score (if comparing to reference set)
   - Text accuracy (for typography prompts - OCR validation)
   - Face detection success rate (for people prompts)

### Category-Specific Performance
- Text rendering accuracy by model
- Human anatomy consistency scores
- Landscape coherence ratings
- Abstract interpretation variance

---

## 5. Experimental Design

### Test Matrix
```python
# Total experiments per model: 5 resolutions × 5 steps × 5 guidance × 50 prompts = 6,250 generations
# With 6 models: 37,500 total generations

# Optimization: Use subset for full matrix
quick_test_prompts = [1, 5, 9, 13, 17]  # 5 representative prompts
full_matrix_test = quick_test_prompts

# Full prompt set only on optimal settings
optimal_settings = {
    "resolution": (768, 768),
    "steps": 24,
    "guidance": 5.0
}
```

### Data Collection Structure
```python
results = {
    "model_id": str,
    "prompt_id": int,
    "category": str,
    "resolution": tuple,
    "steps": int,
    "guidance_scale": float,
    "generation_time": float,
    "vram_peak": float,
    "clip_score": float,
    "text_accuracy": float,  # for text prompts
    "face_detected": bool,   # for people prompts
    "nsfw_flagged": bool,
    "black_image": bool,
    "error": str or None
}
```

---

## 6. Analysis and Visualization Plan

### Primary Visualizations

1. **Resolution Scaling Performance**
   - Line plot: Generation time vs Resolution (per model)
   - Heatmap: Quality score vs Resolution×Model

2. **Steps vs Quality Tradeoff**
   - Scatter plot with Pareto frontier
   - Box plot: Time distribution by step count

3. **Category Performance Radar Chart**
   - Each model's performance across categories
   - Normalized 0-1 scale

4. **Speed-Quality Efficiency Curve**
   - X-axis: Generation time
   - Y-axis: Quality metric
   - Point size: Resolution
   - Color: Model

5. **Whisker Box Plots** (as you mentioned)
   - Generation time distribution per model
   - Separate by resolution
   - Color by model family

### Statistical Analysis
- ANOVA for model performance differences
- Regression analysis for parameter sensitivity
- Standard deviation of generation times
- Confidence intervals for mean performance

---

## 7. Implementation Code Structure

```python
# benchmark_runner.py
import torch
import time
import json
import pandas as pd
from pathlib import Path
import psutil
import GPUtil

class ModelBenchmark:
    def __init__(self, model_configs, prompts, output_dir):
        self.models = model_configs
        self.prompts = prompts
        self.output_dir = Path(output_dir)
        self.results = []
        
    def run_benchmark(self, model_id, prompt, params):
        torch.cuda.empty_cache()
        
        # Monitor GPU
        gpu = GPUtil.getGPUs()[0]
        vram_start = gpu.memoryUsed
        
        start_time = time.perf_counter()
        
        # Generation code here
        image = self.generate(model_id, prompt, params)
        
        end_time = time.perf_counter()
        vram_peak = gpu.memoryUsed
        
        return {
            "generation_time": end_time - start_time,
            "vram_used": vram_peak - vram_start,
            "image": image
        }
    
    def calculate_metrics(self, image, prompt):
        # CLIP score, OCR for text, face detection, etc.
        pass
    
    def save_results(self):
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_dir / "benchmark_results.csv")
        df.to_json(self.output_dir / "benchmark_results.json", orient='records')
```

---

## 8. Key Research Questions to Answer

1. **Which model provides the best speed-to-quality ratio?**
2. **How does resolution scaling affect different architectures?**
3. **What is the optimal step count for each model?**
4. **Which models handle text rendering most accurately?**
5. **How do sparse models (HiDream) compare to dense models in efficiency?**
6. **What is the correlation between guidance scale and quality across models?**
7. **Which model has the most consistent performance across prompt categories?**

---

## 9. Publication Strategy

### Blog Post Structure
1. **Introduction**: The need for comprehensive benchmarking
2. **Methodology**: Hardware setup, model selection rationale
3. **Results**: Interactive charts, key findings
4. **Model Recommendations**: Use-case specific guidance
5. **Technical Deep Dive**: Architecture comparisons
6. **Reproducibility**: Code and data availability

### Key Differentiators
- Using cutting-edge hardware (RTX 5090)
- Comprehensive prompt diversity
- Multiple parameter dimensions
- Category-specific analysis
- Open-source benchmark suite

---

## 10. Expected Insights

Based on preliminary analysis, we expect to find:
- SDXL Lightning will dominate speed metrics but struggle with text
- HiDream I1 will excel at typography but require more VRAM
- FLUX.1-dev will show best overall balance
- Playground v2.5 will have most consistent aesthetic
- Resolution scaling will be non-linear across models
- Optimal steps will vary by 2-3x between models

---

## Appendix: Extended Prompt List

[Continue with remaining 35 prompts across all categories...]

```json
{
  "extended_prompts": [
    {"id": 16, "category": "text", "prompt": "Restaurant receipt showing 'Total: $47.82 - Thank You!'"},
    {"id": 17, "category": "text", "prompt": "Street graffiti spelling 'DREAM BIG' in bubble letters"},
    {"id": 18, "category": "text", "prompt": "Book cover: 'The Art of AI' by S. Johnson, bestseller badge"},
    {"id": 19, "category": "text", "prompt": "License plate reading 'AI-2025' on a futuristic car"},
    {"id": 20, "category": "text", "prompt": "Handwritten note: 'Meet me at sunset - Love, M'"},
    {"id": 21, "category": "people", "prompt": "Diverse team of scientists in a laboratory, all wearing safety goggles"},
    {"id": 22, "category": "people", "prompt": "Ballet dancer mid-leap, spotlight, dramatic shadows"},
    {"id": 23, "category": "people", "prompt": "Construction worker drinking coffee at sunrise, hard hat, reflective vest"},
    {"id": 24, "category": "people", "prompt": "Twin babies laughing in matching outfits"},
    {"id": 25, "category": "people", "prompt": "Street artist painting a mural, paint-splattered clothes, concentration"},
    {"id": 26, "category": "people", "prompt": "Surgeon performing operation, intense focus, medical equipment"},
    {"id": 27, "category": "people", "prompt": "Jazz musician playing saxophone in smoky club, vintage atmosphere"},
    {"id": 28, "category": "people", "prompt": "Yoga instructor in tree pose on beach at dawn"},
    {"id": 29, "category": "people", "prompt": "Chef tossing pizza dough, flour in the air, Italian restaurant"},
    {"id": 30, "category": "people", "prompt": "Astronaut floating in space station, Earth visible through window"},
    {"id": 31, "category": "animals", "prompt": "Hummingbird feeding from red flower, wings in motion"},
    {"id": 32, "category": "animals", "prompt": "Pride of lions resting under acacia tree, African savanna"},
    {"id": 33, "category": "animals", "prompt": "Tropical fish swimming through coral reef, vibrant colors"},
    {"id": 34, "category": "animals", "prompt": "Horse galloping through shallow water, mane flowing"},
    {"id": 35, "category": "animals", "prompt": "Butterfly emerging from chrysalis, transformation moment"},
    {"id": 36, "category": "animals", "prompt": "Wolf howling at full moon, snowy forest"},
    {"id": 37, "category": "animals", "prompt": "Pandas playing in bamboo forest, mother and cub"},
    {"id": 38, "category": "landscape", "prompt": "Northern lights dancing over frozen lake, pine trees silhouette"},
    {"id": 39, "category": "landscape", "prompt": "Tokyo street at night, rain, neon reflections, crowds with umbrellas"},
    {"id": 40, "category": "landscape", "prompt": "Ancient ruins overtaken by jungle, mysterious atmosphere"},
    {"id": 41, "category": "landscape", "prompt": "Lighthouse on cliff during storm, dramatic waves crashing"},
    {"id": 42, "category": "landscape", "prompt": "Lavender fields in Provence, old farmhouse, sunset"},
    {"id": 43, "category": "landscape", "prompt": "Grand Canyon from above, layers of red rock, river below"},
    {"id": 44, "category": "landscape", "prompt": "Venice canal with gondolas, ancient buildings, golden hour"},
    {"id": 45, "category": "landscape", "prompt": "Futuristic Mars colony, domes and solar panels, red landscape"},
    {"id": 46, "category": "abstract", "prompt": "Sound waves visualized as colorful ribbons in space"},
    {"id": 47, "category": "abstract", "prompt": "Human consciousness represented as interconnected galaxies"},
    {"id": 48, "category": "abstract", "prompt": "Mathematics becoming physical: equations transforming into structures"},
    {"id": 49, "category": "abstract", "prompt": "Emotions as weather patterns colliding"},
    {"id": 50, "category": "abstract", "prompt": "Digital decay: pixels dissolving into organic forms"}
  ]
}
```

---

## Implementation Notes

1. **Start with Quick Tests**: Run 5 prompts across all parameters first
2. **Use Caching**: Save generated images for quality analysis
3. **Monitor Temperature**: GPU thermal throttling can affect results
4. **Document Failures**: Track NSFW flags, black images, OOM errors
5. **Version Control**: Track model versions/checksums

This benchmark will position you as a technical authority in image generation model evaluation.
