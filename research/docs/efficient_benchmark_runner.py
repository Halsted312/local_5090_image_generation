#!/usr/bin/env python3
"""
Efficient Image Generation Benchmark Runner
Optimized for 15-hour testing window on RTX 5090
Author: Stephen
Date: November 22, 2025
"""

import torch
import time
import json
import os
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import psutil
import GPUtil
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable TF32 for NVIDIA GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class GPULease:
    """
    File-based GPU lease that yields to backend.
    Backend writes owner=backend; benchmark writes owner=benchmark.
    The backend will wait briefly for the lease to clear; benchmark should only hold
    for one generation, then release and flush CUDA.
    """

    def __init__(
        self,
        owner: str = "benchmark",
        lease_path: str | None = None,
        poll_interval: float = 1.0,
        max_wait_seconds: int = 120,
        stale_after_seconds: int = 600,
    ) -> None:
        env_path = lease_path or os.getenv("GPU_COORD_PATH", "/gpu_coord/status.json")
        self.path = Path(env_path) if env_path else None
        self.owner = owner
        self.poll_interval = poll_interval
        self.max_wait_seconds = max_wait_seconds
        self.stale_after_seconds = stale_after_seconds

    def _read(self) -> tuple[str | None, float | None]:
        if not self.path or not self.path.exists():
            return None, None
        try:
            data = json.loads(self.path.read_text())
            return data.get("owner"), float(data.get("since", 0.0))
        except Exception:
            return None, None

    def _write(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"owner": self.owner, "since": time.time()})
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(payload)
        tmp_path.replace(self.path)

    def acquire(self) -> bool:
        if not self.path:
            return True
        start = time.time()
        while True:
            try:
                # Atomic create; fail if exists
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, json.dumps({"owner": self.owner, "since": time.time()}).encode("utf-8"))
                os.close(fd)
                return True
            except FileExistsError:
                owner, since = self._read()
                now = time.time()
                stale = since is not None and (now - since) > self.stale_after_seconds
                if stale:
                    try:
                        self.path.unlink(missing_ok=True)
                        logger.warning("Removed stale GPU lease (owner=%s, age=%.1fs)", owner, now - since)
                        continue
                    except Exception:
                        pass
                if owner and owner != self.owner:
                    waited = now - start
                    if waited > self.max_wait_seconds:
                        logger.info("Still waiting on GPU lease held by %s for %.1fs", owner, waited)
                    time.sleep(self.poll_interval)
                else:
                    # Same owner lingering; try to reclaim
                    try:
                        self.path.unlink(missing_ok=True)
                    except Exception:
                        time.sleep(self.poll_interval)
            except Exception as exc:  # noqa: BLE001
                logger.warning("GPU lease acquire error: %s", exc)
                time.sleep(self.poll_interval)

    def release(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            owner, _ = self._read()
            if owner != self.owner:
                return
            self.path.unlink(missing_ok=True)
        except Exception:
            pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        except Exception:
            pass
        time.sleep(0.5)  # brief calm period before backend reclaims GPU

@dataclass
class TestConfig:
    """Configuration for efficient testing"""
    # Phase 1: Quick validation (3 models, 5 prompts)
    phase1_models: List[str] = None
    phase1_prompts: List[int] = None
    
    # Phase 2: Focused testing (all models, 20 prompts)
    phase2_prompts: List[int] = None
    
    # Phase 3: Resolution scaling (top 3 models)
    phase3_resolutions: List[Tuple[int, int]] = None
    
    # Fixed parameters for efficiency
    base_resolution: Tuple[int, int] = (512, 512)
    test_tf32: bool = True  # Test TF32 in phase 1 only
    
    def __post_init__(self):
        if self.phase1_models is None:
            self.phase1_models = ["flux_dev", "sdxl_lightning", "sd3_medium"]
        
        if self.phase1_prompts is None:
            # One from each category: text, people, animals, landscapes, abstract
            self.phase1_prompts = [2, 4, 7, 9, 11]
        
        if self.phase2_prompts is None:
            # 4 from each category (20 total)
            self.phase2_prompts = [
                1, 12, 17, 51,      # text
                4, 22, 53, 61,      # people  
                7, 31, 52, 72,      # animals
                9, 38, 54, 66,      # landscapes
                11, 46, 47, 48      # abstract
            ]
        
        if self.phase3_resolutions is None:
            self.phase3_resolutions = [
                (256, 256),
                (512, 512),
                (768, 768),
                (1024, 1024)
            ]

class EfficientBenchmark:
    """Optimized benchmark runner for 15-hour window"""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.output_dir / "benchmark.db"
        
        # Create subdirectories
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        self.config = TestConfig()
        self.results = []
        self.start_time = None
        self.time_budget = timedelta(hours=15)
        
        # Model-specific optimal parameters
        self.model_configs = self._get_model_configs()
        
        # Load prompts
        self.prompts = self._load_prompts()
        self._init_db()
        self._sync_prompts()
        
        # GPU monitoring
        self.gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
        # Coordination lease so backend can wait a few seconds while we finish a single image.
        self.gpu_lease = GPULease(owner="benchmark")
    
    def _get_model_configs(self) -> Dict:
        """Get optimized parameters for each model"""
        return {
            "flux_dev": {
                "name": "FLUX.1-dev",
                "path": "black-forest-labs/FLUX.1-dev",
                "steps": [12, 20, 28, 36],
                "guidance": [3.5, 4.5, 5.5],
                "optimal_steps": 24,
                "optimal_guidance": 4.5
            },
            "sdxl_lightning": {
                "name": "SDXL-Lightning",
                "path": "ByteDance/SDXL-Lightning",
                "steps": [2, 4, 6, 8],
                "guidance": [1.0, 1.5, 2.0],
                "optimal_steps": 4,
                "optimal_guidance": 1.5
            },
            "sd3_medium": {
                "name": "SD3 Medium",
                "path": "stabilityai/stable-diffusion-3-medium",
                "steps": [20, 28, 35, 42],
                "guidance": [5.0, 6.5, 8.0],
                "optimal_steps": 28,
                "optimal_guidance": 6.5
            },
            "hidream": {
                "name": "HiDream-I1",
                "path": "HiDream-ai/HiDream-I1-Full",
                "steps": [12, 18, 24, 30],
                "guidance": [3.0, 4.0, 5.0],
                "optimal_steps": 20,
                "optimal_guidance": 4.0
            },
            "sdxl": {
                "name": "SDXL 1.0",
                "path": "stabilityai/stable-diffusion-xl-base-1.0",
                "steps": [20, 30, 40, 50],
                "guidance": [5.0, 7.0, 9.0],
                "optimal_steps": 30,
                "optimal_guidance": 7.0
            },
            "deepfloyd": {
                "name": "DeepFloyd IF",
                "path": "DeepFloyd/IF-I-XL-v1.0",
                "steps": [50, 75, 100],  # Fewer tests due to slowness
                "guidance": [5.0, 7.5],
                "optimal_steps": 75,
                "optimal_guidance": 6.0
            },
            "realvis": {
                "name": "RealVisXL",
                "path": "SG161222/Realistic_Vision_V5.1_noVAE",
                "steps": [20, 26, 32],
                "guidance": [4.5, 5.5, 6.5],
                "optimal_steps": 26,
                "optimal_guidance": 5.5
            }
        }
    
    def _load_prompts(self) -> List[Dict]:
        """Load benchmark prompts"""
        prompt_file = Path("benchmark_prompts_v2.json")
        if prompt_file.exists():
            with open(prompt_file, 'r') as f:
                data = json.load(f)
                return data['prompts']
        else:
            # Minimal fallback prompts
            return [
                {"id": i, "prompt": f"Test prompt {i}", "category": "test"}
                for i in range(1, 21)
            ]

    def _init_db(self) -> None:
        """Initialize SQLite schema for results and prompts."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                phase TEXT,
                timestamp TEXT,
                model_id TEXT,
                model_name TEXT,
                prompt_id INTEGER,
                prompt_category TEXT,
                resolution_w INTEGER,
                resolution_h INTEGER,
                steps INTEGER,
                guidance REAL,
                use_tf32 INTEGER,
                generation_time REAL,
                time_per_step REAL,
                time_per_megapixel REAL,
                vram_used_mb REAL,
                clip_score REAL,
                aesthetic_score REAL,
                success INTEGER
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prompts (
                prompt_id INTEGER PRIMARY KEY,
                category TEXT,
                complexity TEXT,
                prompt TEXT
            )
            """
        )
        self.conn.commit()

    def _sync_prompts(self) -> None:
        """Ensure prompts table is populated."""
        if not hasattr(self, "conn"):
            return
        cur = self.conn.cursor()
        rows = [
            (p.get("id"), p.get("category"), p.get("complexity"), p.get("prompt"))
            for p in self.prompts
        ]
        cur.executemany(
            """
            INSERT OR IGNORE INTO prompts (prompt_id, category, complexity, prompt)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def _record_result(self, result: Dict) -> None:
        """Persist a single result row to SQLite."""
        if not hasattr(self, "conn"):
            return
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO runs (
                phase, timestamp, model_id, model_name, prompt_id, prompt_category,
                resolution_w, resolution_h, steps, guidance, use_tf32,
                generation_time, time_per_step, time_per_megapixel,
                vram_used_mb, clip_score, aesthetic_score, success
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.get("phase"),
                result.get("timestamp"),
                result.get("model_id"),
                result.get("model_name"),
                result.get("prompt_id"),
                result.get("prompt_category"),
                result.get("resolution")[0] if result.get("resolution") else None,
                result.get("resolution")[1] if result.get("resolution") else None,
                result.get("steps"),
                result.get("guidance"),
                int(bool(result.get("use_tf32"))),
                result.get("generation_time"),
                result.get("time_per_step"),
                result.get("time_per_megapixel"),
                result.get("vram_used_mb"),
                result.get("clip_score"),
                result.get("aesthetic_score"),
                int(bool(result.get("success"))),
            ),
        )
        self.conn.commit()
    
    def estimate_time(self, phase: str) -> timedelta:
        """Estimate time for each phase"""
        if phase == "phase1":
            # 3 models × 5 prompts × 2 TF32 × 4 steps × 3 guidance
            n_tests = 3 * 5 * 2 * 4 * 3
            avg_time = 8  # seconds per generation
            return timedelta(seconds=n_tests * avg_time)
        
        elif phase == "phase2":
            # 7 models × 20 prompts × varying steps/guidance
            n_tests = 0
            for model_id in self.model_configs:
                config = self.model_configs[model_id]
                n_tests += 20 * len(config['steps']) * len(config['guidance'])
            avg_time = 10
            return timedelta(seconds=n_tests * avg_time)
        
        elif phase == "phase3":
            # 3 models × 10 prompts × 4 resolutions
            n_tests = 3 * 10 * 4
            avg_time = 12
            return timedelta(seconds=n_tests * avg_time)

    def generate_with_lease(self, fn, *args, **kwargs):
        """
        Wrap a single generation under the GPU lease.
        `fn` should perform one image generation and return metrics/image path.
        """
        with self.gpu_lease:
            return fn(*args, **kwargs)
    
    def run_phase1_validation(self):
        """Phase 1: Quick validation with 3 models"""
        logger.info("="*60)
        logger.info("PHASE 1: Quick Validation")
        logger.info(f"Models: {self.config.phase1_models}")
        logger.info(f"Prompts: {len(self.config.phase1_prompts)}")
        logger.info(f"Estimated time: {self.estimate_time('phase1')}")
        logger.info("="*60)
        
        phase1_results = []
        
        with tqdm(total=360, desc="Phase 1 Progress") as pbar:
            for model_id in self.config.phase1_models:
                model_config = self.model_configs[model_id]
                
                for prompt_id in self.config.phase1_prompts:
                    prompt = self.prompts[prompt_id - 1]
                    
                    # Test with and without TF32
                    for use_tf32 in [True, False]:
                        torch.backends.cuda.matmul.allow_tf32 = use_tf32
                        torch.backends.cudnn.allow_tf32 = use_tf32
                        
                        for steps in model_config['steps']:
                            for guidance in model_config['guidance']:
                                result = self._run_single_test(
                                    model_id=model_id,
                                    model_name=model_config['name'],
                                    prompt=prompt,
                                    resolution=self.config.base_resolution,
                                    steps=steps,
                                    guidance=guidance,
                                    use_tf32=use_tf32,
                                    phase="phase1"
                                )
                                
                                phase1_results.append(result)
                                self.results.append(result)
                                self._record_result(result)
                                pbar.update(1)
                                
                                # Check time budget
                                if self._check_time_limit():
                                    logger.warning("Time budget exceeded!")
                                    self._save_checkpoint("phase1_incomplete")
                                    return phase1_results
        
        # Re-enable TF32 for remaining tests
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Analyze Phase 1 results
        self._analyze_phase1(phase1_results)
        self._save_checkpoint("phase1_complete")
        
        return phase1_results
    
    def run_phase2_focused(self):
        """Phase 2: Focused testing with all models"""
        logger.info("="*60)
        logger.info("PHASE 2: Focused Deep Dive")
        logger.info(f"Models: All 7")
        logger.info(f"Prompts: {len(self.config.phase2_prompts)}")
        logger.info(f"Estimated time: {self.estimate_time('phase2')}")
        logger.info("="*60)
        
        phase2_results = []
        
        # Calculate total tests for progress bar
        total_tests = sum(
            len(self.config.phase2_prompts) * 
            len(config['steps']) * 
            len(config['guidance'])
            for config in self.model_configs.values()
        )
        
        with tqdm(total=total_tests, desc="Phase 2 Progress") as pbar:
            for model_id, model_config in self.model_configs.items():
                logger.info(f"Testing {model_config['name']}...")
                
                for prompt_id in self.config.phase2_prompts:
                    prompt = self.prompts[prompt_id - 1]
                    
                    for steps in model_config['steps']:
                        for guidance in model_config['guidance']:
                            result = self._run_single_test(
                                model_id=model_id,
                                model_name=model_config['name'],
                                prompt=prompt,
                                resolution=self.config.base_resolution,
                                steps=steps,
                                guidance=guidance,
                                use_tf32=True,
                                phase="phase2"
                            )
                            
                            phase2_results.append(result)
                            self.results.append(result)
                            self._record_result(result)
                            pbar.update(1)
                            
                            # Check time budget
                            if self._check_time_limit():
                                logger.warning("Time budget exceeded!")
                                self._save_checkpoint("phase2_incomplete")
                                return phase2_results
        
        # Analyze Phase 2 results
        self._analyze_phase2(phase2_results)
        self._save_checkpoint("phase2_complete")
        
        return phase2_results
    
    def run_phase3_resolution(self, top_models: List[str] = None):
        """Phase 3: Resolution scaling with top 3 models"""
        
        # If top models not specified, analyze Phase 2 to find them
        if top_models is None:
            top_models = self._identify_top_models()
        
        logger.info("="*60)
        logger.info("PHASE 3: Resolution Scaling")
        logger.info(f"Models: {top_models}")
        logger.info(f"Resolutions: {self.config.phase3_resolutions}")
        logger.info(f"Estimated time: {self.estimate_time('phase3')}")
        logger.info("="*60)
        
        phase3_results = []
        
        # Use first 10 prompts (2 from each category)
        test_prompts = [1, 12, 4, 22, 7, 31, 9, 38, 11, 46]
        
        total_tests = len(top_models) * len(test_prompts) * len(self.config.phase3_resolutions)
        
        with tqdm(total=total_tests, desc="Phase 3 Progress") as pbar:
            for model_id in top_models:
                model_config = self.model_configs[model_id]
                
                for prompt_id in test_prompts:
                    prompt = self.prompts[prompt_id - 1]
                    
                    for resolution in self.config.phase3_resolutions:
                        result = self._run_single_test(
                            model_id=model_id,
                            model_name=model_config['name'],
                            prompt=prompt,
                            resolution=resolution,
                            steps=model_config['optimal_steps'],
                            guidance=model_config['optimal_guidance'],
                            use_tf32=True,
                            phase="phase3"
                        )
                        
                        phase3_results.append(result)
                        self.results.append(result)
                        self._record_result(result)
                        pbar.update(1)
                        
                        # Check time budget
                        if self._check_time_limit():
                            logger.warning("Time budget exceeded!")
                            self._save_checkpoint("phase3_incomplete")
                            return phase3_results
        
        # Analyze Phase 3 results
        self._analyze_phase3(phase3_results)
        self._save_checkpoint("phase3_complete")
        
        return phase3_results
    
    def _run_single_test(
        self,
        model_id: str,
        model_name: str,
        prompt: Dict,
        resolution: Tuple[int, int],
        steps: int,
        guidance: float,
        use_tf32: bool,
        phase: str
    ) -> Dict:
        """Run a single benchmark test"""
        
        def _do_generation() -> Dict:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Record start metrics
            vram_start, _ = self._get_gpu_metrics() if self.gpu else (0, 0)
            
            start_time = time.perf_counter()
            
            # ============================================
            # PLACEHOLDER: Actual generation would go here
            # ============================================
            
            # Simulate generation with appropriate delay
            if model_id == "sdxl_lightning":
                delay = np.random.uniform(0.8, 1.5)  # Fast model
            elif model_id == "deepfloyd":
                delay = np.random.uniform(10, 15)  # Slow model
            else:
                delay = np.random.uniform(3, 8)  # Medium models
            
            # Scale delay by resolution
            resolution_factor = (resolution[0] * resolution[1]) / (512 * 512)
            delay *= np.sqrt(resolution_factor)
            
            # Scale by steps
            step_factor = steps / 30
            delay *= step_factor
            
            time.sleep(min(delay, 15))  # Cap at 15 seconds for testing
            
            # End timing
            end_time = time.perf_counter()
            generation_time = end_time - start_time
            
            # Get final metrics
            vram_end, _ = self._get_gpu_metrics() if self.gpu else (0, 0)
            
            # Simulate quality metrics
            clip_score = np.random.uniform(0.25, 0.35) + (0.01 * steps / 30)
            aesthetic_score = np.random.uniform(5.0, 8.0) + (0.5 * guidance / 7)
            
            return {
                "phase": phase,
                "timestamp": datetime.now().isoformat(),
                "model_id": model_id,
                "model_name": model_name,
                "prompt_id": prompt['id'],
                "prompt_category": prompt['category'],
                "resolution": resolution,
                "resolution_str": f"{resolution[0]}x{resolution[1]}",
                "steps": steps,
                "guidance": guidance,
                "use_tf32": use_tf32,
                "generation_time": generation_time,
                "time_per_step": generation_time / steps,
                "time_per_megapixel": generation_time / (resolution[0] * resolution[1] / 1e6),
                "vram_used_mb": (vram_end - vram_start) * 1024,
                "clip_score": clip_score,
                "aesthetic_score": aesthetic_score,
                "success": True
            }

        return self.generate_with_lease(_do_generation)
    
    def _get_gpu_metrics(self) -> Tuple[float, float]:
        """Get GPU memory usage"""
        if self.gpu:
            self.gpu = GPUtil.getGPUs()[0]
            return self.gpu.memoryUsed, self.gpu.memoryTotal
        return 0.0, 0.0
    
    def _check_time_limit(self) -> bool:
        """Check if we're approaching time budget"""
        if self.start_time is None:
            return False
        
        elapsed = datetime.now() - self.start_time
        return elapsed >= self.time_budget * 0.95  # Stop at 95% of budget
    
    def _identify_top_models(self) -> List[str]:
        """Identify top 3 models from Phase 2 results"""
        if not self.results:
            return ["flux_dev", "sd3_medium", "sdxl"]
        
        df = pd.DataFrame(self.results)
        phase2_df = df[df['phase'] == 'phase2']
        
        if phase2_df.empty:
            return ["flux_dev", "sd3_medium", "sdxl"]
        
        # Score based on speed and quality
        model_scores = phase2_df.groupby('model_id').agg({
            'generation_time': 'mean',
            'clip_score': 'mean',
            'aesthetic_score': 'mean'
        })
        
        # Normalize and combine scores
        model_scores['speed_score'] = 1.0 / (1.0 + model_scores['generation_time'])
        model_scores['quality_score'] = (
            model_scores['clip_score'] * 0.5 + 
            model_scores['aesthetic_score'] / 10 * 0.5
        )
        model_scores['total_score'] = (
            model_scores['speed_score'] * 0.4 + 
            model_scores['quality_score'] * 0.6
        )
        
        top_3 = model_scores.nlargest(3, 'total_score').index.tolist()
        logger.info(f"Top 3 models identified: {top_3}")
        
        return top_3
    
    def _analyze_phase1(self, results: List[Dict]):
        """Analyze Phase 1 results"""
        df = pd.DataFrame(results)
        
        # TF32 impact analysis
        tf32_impact = df.groupby(['model_id', 'use_tf32'])['generation_time'].mean().unstack()
        tf32_speedup = (tf32_impact[False] - tf32_impact[True]) / tf32_impact[False] * 100
        
        logger.info("\n" + "="*40)
        logger.info("PHASE 1 ANALYSIS")
        logger.info("="*40)
        logger.info("\nTF32 Speedup by Model:")
        for model, speedup in tf32_speedup.items():
            logger.info(f"  {model}: {speedup:.1f}% faster with TF32")
        
        # Step efficiency analysis
        logger.info("\nOptimal Steps Analysis:")
        for model in self.config.phase1_models:
            model_df = df[df['model_id'] == model]
            step_quality = model_df.groupby('steps')['clip_score'].mean()
            step_time = model_df.groupby('steps')['generation_time'].mean()
            
            # Find best efficiency (quality per second)
            efficiency = step_quality / step_time
            best_steps = efficiency.idxmax()
            
            logger.info(f"  {model}: {best_steps} steps optimal")
    
    def _analyze_phase2(self, results: List[Dict]):
        """Analyze Phase 2 results"""
        df = pd.DataFrame(results)
        
        logger.info("\n" + "="*40)
        logger.info("PHASE 2 ANALYSIS")
        logger.info("="*40)
        
        # Model performance summary
        model_summary = df.groupby('model_name').agg({
            'generation_time': 'mean',
            'clip_score': 'mean',
            'aesthetic_score': 'mean'
        }).round(2)
        
        logger.info("\nModel Performance Summary:")
        print(model_summary)
        
        # Category performance
        category_performance = df.groupby(['model_name', 'prompt_category']).agg({
            'generation_time': 'mean',
            'clip_score': 'mean'
        })
        
        logger.info("\nBest Model by Category:")
        for category in df['prompt_category'].unique():
            cat_df = df[df['prompt_category'] == category]
            best_model = cat_df.groupby('model_name')['clip_score'].mean().idxmax()
            logger.info(f"  {category}: {best_model}")
    
    def _analyze_phase3(self, results: List[Dict]):
        """Analyze Phase 3 results"""
        df = pd.DataFrame(results)
        
        logger.info("\n" + "="*40)
        logger.info("PHASE 3 ANALYSIS")
        logger.info("="*40)
        
        # Resolution scaling analysis
        resolution_scaling = df.groupby(['model_name', 'resolution_str']).agg({
            'generation_time': 'mean',
            'time_per_megapixel': 'mean'
        })
        
        logger.info("\nResolution Scaling Efficiency:")
        print(resolution_scaling)
        
        # Find linear vs super-linear scaling
        for model in df['model_name'].unique():
            model_df = df[df['model_name'] == model]
            times = model_df.groupby('resolution_str')['generation_time'].mean()
            
            # Check if scaling is linear
            if len(times) >= 2:
                scaling_factor = times.iloc[-1] / times.iloc[0]
                pixels_factor = (1024 * 1024) / (256 * 256)
                
                if scaling_factor < pixels_factor * 0.9:
                    logger.info(f"  {model}: Sub-linear scaling (efficient)")
                elif scaling_factor > pixels_factor * 1.1:
                    logger.info(f"  {model}: Super-linear scaling (inefficient)")
                else:
                    logger.info(f"  {model}: Linear scaling")
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save intermediate results"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        # Save CSV
        csv_file = self.checkpoints_dir / f"{checkpoint_name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        df.to_csv(csv_file, index=False)
        
        # Save JSON
        json_file = self.checkpoints_dir / f"{checkpoint_name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Checkpoint saved: {csv_file}")
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        if not self.results:
            logger.warning("No results to report")
            return
        
        df = pd.DataFrame(self.results)
        
        report_file = self.output_dir / f"final_report_{datetime.now():%Y%m%d_%H%M%S}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Benchmark Results Summary\n\n")
            f.write(f"Total tests completed: {len(df)}\n")
            f.write(f"Total time elapsed: {datetime.now() - self.start_time}\n\n")
            
            # Overall rankings
            f.write("## Model Rankings\n\n")
            
            # Speed ranking
            speed_ranking = df.groupby('model_name')['generation_time'].mean().sort_values()
            f.write("### Fastest Models (512x512)\n")
            for i, (model, time) in enumerate(speed_ranking.items(), 1):
                f.write(f"{i}. {model}: {time:.2f}s\n")
            
            # Quality ranking
            if 'clip_score' in df.columns:
                quality_ranking = df.groupby('model_name')['clip_score'].mean().sort_values(ascending=False)
                f.write("\n### Highest Quality Models\n")
                for i, (model, score) in enumerate(quality_ranking.items(), 1):
                    f.write(f"{i}. {model}: {score:.3f}\n")
            
            # TF32 Impact
            if len(df['use_tf32'].unique()) > 1:
                f.write("\n## TF32 Impact\n")
                tf32_impact = df.groupby(['model_name', 'use_tf32'])['generation_time'].mean()
                f.write(f"Average speedup with TF32: {tf32_impact.mean():.1%}\n")
            
            # Parameter optimization
            f.write("\n## Optimal Parameters\n")
            for model in df['model_name'].unique():
                model_df = df[df['model_name'] == model]
                
                # Find best efficiency point
                model_df['efficiency'] = model_df['clip_score'] / model_df['generation_time']
                best = model_df.loc[model_df['efficiency'].idxmax()]
                
                f.write(f"\n### {model}\n")
                f.write(f"- Best steps: {best['steps']}\n")
                f.write(f"- Best guidance: {best['guidance']:.1f}\n")
                f.write(f"- Generation time: {best['generation_time']:.2f}s\n")
        
        logger.info(f"Final report saved: {report_file}")
    
    def run_complete_benchmark(self):
        """Run all three phases within time budget"""
        self.start_time = datetime.now()
        logger.info(f"Benchmark started at {self.start_time}")
        logger.info(f"Time budget: {self.time_budget}")
        
        # Phase 1: Quick validation
        phase1_results = self.run_phase1_validation()
        logger.info(f"Phase 1 completed: {len(phase1_results)} tests")
        
        # Check remaining time
        elapsed = datetime.now() - self.start_time
        if elapsed >= self.time_budget * 0.3:
            logger.info("Phase 1 took longer than expected, adjusting...")
        
        # Phase 2: Focused testing
        phase2_results = self.run_phase2_focused()
        logger.info(f"Phase 2 completed: {len(phase2_results)} tests")
        
        # Check remaining time
        elapsed = datetime.now() - self.start_time
        if elapsed < self.time_budget * 0.85:
            # Phase 3: Resolution scaling
            phase3_results = self.run_phase3_resolution()
            logger.info(f"Phase 3 completed: {len(phase3_results)} tests")
        else:
            logger.info("Skipping Phase 3 due to time constraints")
        
        # Generate final report
        self.generate_final_report()
        
        total_time = datetime.now() - self.start_time
        logger.info(f"Benchmark completed in {total_time}")
        logger.info(f"Total tests: {len(self.results)}")

def main():
    """Main entry point"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║   Efficient Image Generation Benchmark (15-hour budget)  ║
    ║          Optimized for NVIDIA RTX 5090                   ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    benchmark = EfficientBenchmark()
    
    print("\nBenchmark Strategy:")
    print("Phase 1: Quick validation (3 models, TF32 testing) - ~2 hours")
    print("Phase 2: Focused testing (7 models, optimal params) - ~10 hours")  
    print("Phase 3: Resolution scaling (top 3 models) - ~3 hours")
    print("\nTotal estimated time: 15 hours")
    
    response = input("\nStart complete benchmark? (y/n): ")
    if response.lower() == 'y':
        benchmark.run_complete_benchmark()
    else:
        print("Benchmark cancelled")

if __name__ == "__main__":
    main()
