I'll create a comprehensive technical implementation document that combines all three responses into a unified system architecture for your backend coding agent. This will include the multi-model routing, prank system, database design, and all storage requirements.

## Comprehensive Backend Implementation Guide for Multi-Model AI Image Generator with Prank System

### System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Replit)                        │
│                  https://promptpics.ai                      │
├─────────────────────────────────────────────────────────────┤
│                         ↓ HTTPS                             │
├─────────────────────────────────────────────────────────────┤
│                   Backend API (FastAPI)                     │
│                https://app.promptpics.ai                    │
│                  (ngrok → localhost:7999)                   │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┬──────────────┬─────────────────────────┐ │
│  │ Router LLM   │ Image Models │  Storage & Database      │ │
│  │              │              │                          │ │
│  │ - Qwen/Llama │ - FLUX.1-dev │ - PostgreSQL/SQLite      │ │
│  │ - JSON       │ - SD3-Medium │ - Image Storage (3TB)    │ │
│  │   routing    │ - RealVisXL  │ - JSONL Logs             │ │
│  │              │ - Logo-SDXL  │ - Thumbnails + Full      │ │
│  └──────────────┴──────────────┴─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Phase 1: Enhanced Database Schema & Storage Architecture

```python
# backend/app/database/models.py
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

from sqlalchemy import (
    Column, String, Text, DateTime, Float, Integer, Boolean,
    ForeignKey, JSON, Enum, UniqueConstraint, Index, LargeBinary
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref

Base = declarative_base()

class GenerationStatus(enum.Enum):
    """Status tracking for generation lifecycle"""
    PENDING = "pending"
    ROUTING = "routing"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    PRANK_MATCHED = "prank_matched"

class ImageVersionType(enum.Enum):
    """Track different versions of images"""
    ORIGINAL = "original"
    EDITED = "edited"
    THUMBNAIL = "thumbnail"
    WATERMARKED = "watermarked"

class PrankSet(Base):
    """
    Represents a collection of prank triggers.
    - builder_url: The secret URL for the creator to manage pranks
    - share_url: The public URL to share with prank victims
    """
    __tablename__ = "prank_sets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    builder_slug = Column(String(8), unique=True, nullable=False, index=True)  # e.g., "a7x9q2lm"
    share_slug = Column(String(8), unique=True, nullable=False, index=True)    # e.g., "k3n5p8wx"
    
    title = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    
    # Tracking
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
    last_accessed = Column(DateTime(timezone=True), nullable=True)
    access_count = Column(Integer, default=0)
    
    # Settings
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    max_uses = Column(Integer, nullable=True)  # Optional usage limit
    
    # Analytics
    total_triggers = Column(Integer, default=0)
    successful_pranks = Column(Integer, default=0)
    
    # Relationships
    triggers = relationship("PrankTrigger", back_populates="prank_set", cascade="all, delete-orphan")
    generations = relationship("Generation", back_populates="prank_set")
    
    # Metadata
    metadata_json = Column(JSON, default=dict)  # Extensible metadata

class PrankTrigger(Base):
    """Individual prank trigger with associated images"""
    __tablename__ = "prank_triggers"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prank_set_id = Column(UUID(as_uuid=True), ForeignKey("prank_sets.id"), nullable=False)
    
    # Trigger configuration
    trigger_prompt = Column(Text, nullable=False)  # The prompt that triggers this prank
    prompt_embedding = Column(JSON, nullable=True)  # For similarity matching
    similarity_threshold = Column(Float, default=0.85)  # How closely prompt must match
    
    # Image storage references
    image_versions = Column(JSON, default=dict)  # {"original": "path", "v1": "path", "v2": "path"}
    current_version = Column(String(50), default="original")
    thumbnail_path = Column(String(500), nullable=True)
    
    # Tracking
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
    trigger_count = Column(Integer, default=0)
    last_triggered = Column(DateTime(timezone=True), nullable=True)
    
    # Priority and matching rules
    priority = Column(Integer, default=0)  # Higher priority triggers match first
    exact_match_only = Column(Boolean, default=False)
    case_sensitive = Column(Boolean, default=False)
    
    # Relationships
    prank_set = relationship("PrankSet", back_populates="triggers")
    images = relationship("PrankImage", back_populates="trigger", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_trigger_prompt_search", "trigger_prompt"),
        Index("idx_prank_set_priority", "prank_set_id", "priority"),
    )

class PrankImage(Base):
    """Store multiple versions of prank images"""
    __tablename__ = "prank_images"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trigger_id = Column(UUID(as_uuid=True), ForeignKey("prank_triggers.id"), nullable=False)
    
    version_number = Column(Integer, nullable=False)  # 1, 2, 3...
    version_type = Column(Enum(ImageVersionType), nullable=False)
    
    # Storage
    file_path = Column(String(500), nullable=False)
    thumbnail_blob = Column(LargeBinary, nullable=True)  # Small thumbnail in DB for speed
    file_size_bytes = Column(Integer, nullable=True)
    
    # Image metadata
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    format = Column(String(10), nullable=True)  # PNG, JPEG, WEBP
    
    # Editing history
    edit_description = Column(Text, nullable=True)
    edited_from_version = Column(Integer, nullable=True)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    trigger = relationship("PrankTrigger", back_populates="images")
    
    __table_args__ = (
        UniqueConstraint("trigger_id", "version_number", "version_type"),
    )

class Generation(Base):
    """Log every single generation request"""
    __tablename__ = "generations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Request data
    prompt = Column(Text, nullable=False)
    prompt_embedding = Column(JSON, nullable=True)  # For analysis
    
    # Routing decision (complete JSON for training)
    routing_decision = Column(JSON, nullable=False)  # Full RoutingDecision object
    chosen_model = Column(String(50), nullable=False)
    model_scores = Column(JSON, nullable=True)  # Detailed scoring
    
    # Generation parameters
    num_inference_steps = Column(Integer)
    guidance_scale = Column(Float)
    width = Column(Integer)
    height = Column(Integer)
    seed = Column(Integer, nullable=True)
    
    # Results
    status = Column(Enum(GenerationStatus), nullable=False)
    image_path = Column(String(500), nullable=True)
    thumbnail_path = Column(String(500), nullable=True)
    generation_time_ms = Column(Integer, nullable=True)
    
    # Prank association
    prank_set_id = Column(UUID(as_uuid=True), ForeignKey("prank_sets.id"), nullable=True)
    was_prank = Column(Boolean, default=False)
    matched_trigger_id = Column(UUID(as_uuid=True), ForeignKey("prank_triggers.id"), nullable=True)
    
    # User tracking (anonymous)
    session_id = Column(String(100), nullable=True)  # Browser session
    ip_hash = Column(String(64), nullable=True)  # Hashed IP for rate limiting
    user_agent = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Quality metrics (for training)
    user_rating = Column(Integer, nullable=True)  # 1-5 stars
    was_regenerated = Column(Boolean, default=False)
    
    # Relationships
    prank_set = relationship("PrankSet", back_populates="generations")
    
    __table_args__ = (
        Index("idx_generation_created", "created_at"),
        Index("idx_generation_session", "session_id"),
        Index("idx_generation_model", "chosen_model"),
    )

class RoutingLog(Base):
    """Detailed routing decisions for training"""
    __tablename__ = "routing_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    generation_id = Column(UUID(as_uuid=True), ForeignKey("generations.id"), nullable=False)
    
    # Complete routing context
    prompt = Column(Text, nullable=False)
    router_model_used = Column(String(100), nullable=False)
    router_prompt_template = Column(Text, nullable=True)
    
    # Decision data
    raw_llm_output = Column(Text, nullable=True)
    parsed_json = Column(JSON, nullable=False)
    
    # Scoring details
    model_candidates = Column(JSON, nullable=False)  # Array of {model, score, reason}
    chosen_model = Column(String(50), nullable=False)
    confidence_score = Column(Float, nullable=True)
    
    # Performance
    routing_time_ms = Column(Integer, nullable=True)
    
    # Training labels
    human_verified = Column(Boolean, default=False)
    correct_choice = Column(Boolean, nullable=True)
    preferred_model = Column(String(50), nullable=True)  # If human disagrees
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    __table_args__ = (
        Index("idx_routing_model", "chosen_model"),
        Index("idx_routing_verification", "human_verified", "correct_choice"),
    )
```

### Phase 2: Storage Management System

```python
# backend/app/storage/manager.py
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from PIL import Image
import io
import base64

class StorageManager:
    """
    Centralized storage management for images and logs.
    Handles thumbnails, versioning, and efficient retrieval.
    """
    
    def __init__(self, base_path: str = "/data/promptpics"):
        self.base_path = Path(base_path)
        self.ensure_directory_structure()
        
        # Directory structure
        self.originals_dir = self.base_path / "originals"
        self.thumbnails_dir = self.base_path / "thumbnails"
        self.pranks_dir = self.base_path / "pranks"
        self.generations_dir = self.base_path / "generations"
        self.logs_dir = self.base_path / "logs"
        
        # JSONL logs for redundancy and quick analysis
        self.generation_log = self.logs_dir / "generations.jsonl"
        self.routing_log = self.logs_dir / "routing_decisions.jsonl"
        self.prank_log = self.logs_dir / "prank_activity.jsonl"
    
    def ensure_directory_structure(self):
        """Create all required directories"""
        dirs = [
            "originals", "thumbnails", "pranks", "generations",
            "logs", "temp", "archive", "training_data"
        ]
        for dir_name in dirs:
            (self.base_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    def save_generation(
        self,
        image: Image.Image,
        prompt: str,
        routing_decision: Dict[str, Any],
        generation_params: Dict[str, Any],
        prank_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Save a generated image with all metadata and create thumbnail.
        Returns paths and metadata for database storage.
        """
        gen_id = uuid.uuid4().hex
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Save full image
        image_filename = f"{gen_id}.png"
        image_path = self.generations_dir / image_filename
        image.save(image_path, format="PNG", optimize=True)
        
        # Create and save thumbnail (256x256)
        thumbnail = self.create_thumbnail(image, size=(256, 256))
        thumb_filename = f"{gen_id}_thumb.webp"
        thumb_path = self.thumbnails_dir / thumb_filename
        thumbnail.save(thumb_path, format="WEBP", quality=85)
        
        # Create thumbnail blob for DB storage (even smaller, 64x64)
        micro_thumb = self.create_thumbnail(image, size=(64, 64))
        thumb_blob = self.image_to_blob(micro_thumb, format="WEBP", quality=70)
        
        # Prepare metadata record
        record = {
            "id": gen_id,
            "timestamp": timestamp,
            "prompt": prompt,
            "routing_decision": routing_decision,
            "generation_params": generation_params,
            "paths": {
                "full_image": str(image_path),
                "thumbnail": str(thumb_path)
            },
            "image_metadata": {
                "width": image.width,
                "height": image.height,
                "format": "PNG",
                "file_size": image_path.stat().st_size
            },
            "prank_info": prank_info,
            "thumbnail_blob": base64.b64encode(thumb_blob).decode('utf-8')
        }
        
        # Log to JSONL for redundancy
        self.append_jsonl(self.generation_log, record)
        
        # Log routing decision separately for training
        routing_record = {
            "generation_id": gen_id,
            "timestamp": timestamp,
            "prompt": prompt,
            **routing_decision
        }
        self.append_jsonl(self.routing_log, routing_record)
        
        return record
    
    def save_prank_image(
        self,
        trigger_id: str,
        image: Image.Image,
        version_number: int = 1,
        edit_description: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Save a prank image with versioning support.
        Returns paths for all saved versions.
        """
        trigger_dir = self.pranks_dir / trigger_id
        trigger_dir.mkdir(exist_ok=True)
        
        # Save original
        original_path = trigger_dir / f"v{version_number}_original.png"
        image.save(original_path, format="PNG")
        
        # Save web-optimized version
        web_path = trigger_dir / f"v{version_number}_web.webp"
        image.save(web_path, format="WEBP", quality=90)
        
        # Save thumbnail
        thumbnail = self.create_thumbnail(image, size=(512, 512))
        thumb_path = trigger_dir / f"v{version_number}_thumb.webp"
        thumbnail.save(thumb_path, format="WEBP", quality=85)
        
        # Create versions manifest
        manifest_path = trigger_dir / "versions.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        else:
            manifest = {"versions": []}
        
        version_info = {
            "version": version_number,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "paths": {
                "original": str(original_path),
                "web": str(web_path),
                "thumbnail": str(thumb_path)
            },
            "edit_description": edit_description,
            "metadata": {
                "width": image.width,
                "height": image.height,
                "file_sizes": {
                    "original": original_path.stat().st_size,
                    "web": web_path.stat().st_size,
                    "thumbnail": thumb_path.stat().st_size
                }
            }
        }
        
        manifest["versions"].append(version_info)
        manifest["current_version"] = version_number
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return version_info["paths"]
    
    def create_thumbnail(self, image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """Create a thumbnail preserving aspect ratio"""
        img_copy = image.copy()
        img_copy.thumbnail(size, Image.Resampling.LANCZOS)
        return img_copy
    
    def image_to_blob(self, image: Image.Image, format: str = "WEBP", quality: int = 85) -> bytes:
        """Convert PIL Image to bytes for database storage"""
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=quality)
        return buffer.getvalue()
    
    def append_jsonl(self, filepath: Path, record: Dict[str, Any]):
        """Append a record to JSONL file for redundancy"""
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
    
    def get_image_hash(self, image: Image.Image) -> str:
        """Generate perceptual hash for duplicate detection"""
        # Simple implementation - can be enhanced with pHash
        img_bytes = image.tobytes()
        return hashlib.sha256(img_bytes).hexdigest()[:16]
    
    def cleanup_old_files(self, days: int = 30):
        """Archive or delete old files based on retention policy"""
        # Implementation for cleaning up old temporary files
        pass
```

### Phase 3: Enhanced Multi-Model Pipeline with Full Routing

```python
# backend/app/models/orchestrator.py
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import torch
from PIL import Image

from .model_registry import ModelRegistry
from .router import RouterLLM, RoutingDecision
from .image_models import (
    generate_with_flux,
    generate_with_sd3,
    generate_with_realvis,
    generate_with_logo_sdxl
)
from ..storage.manager import StorageManager
from ..database.models import Generation, GenerationStatus, PrankSet

logger = logging.getLogger(__name__)

@dataclass
class GenerationRequest:
    """Comprehensive generation request with all parameters"""
    prompt: str
    prank_slug: Optional[str] = None
    num_inference_steps: int = 28
    guidance_scale: float = 4.0
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = None
    engine_override: Optional[str] = None  # Force specific engine
    session_id: Optional[str] = None
    user_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GenerationResult:
    """Complete generation result with all tracking data"""
    image: Image.Image
    generation_id: str
    routing_decision: RoutingDecision
    was_prank: bool
    matched_trigger_id: Optional[str]
    generation_time_ms: int
    storage_paths: Dict[str, str]
    thumbnail_blob: bytes

class ModelOrchestrator:
    """
    Central orchestration engine that handles:
    - Prank detection and matching
    - Model routing decisions
    - Generation execution
    - Complete logging and storage
    """
    
    def __init__(self, storage_manager: StorageManager, db_session):
        self.storage = storage_manager
        self.db = db_session
        self.router = RouterLLM()
        self.registry = ModelRegistry()
        
        # Model execution map
        self.model_executors = {
            "flux_dev": generate_with_flux,
            "sd3_medium": generate_with_sd3,
            "realvis_xl": generate_with_realvis,
            "logo_sdxl": generate_with_logo_sdxl
        }
        
        # Prank matching cache (refresh periodically)
        self._prank_cache = {}
        self._refresh_prank_cache()
    
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Main generation pipeline with full orchestration.
        """
        start_time = time.time()
        
        # Step 1: Check for prank match if slug provided
        prank_match = None
        if request.prank_slug:
            prank_match = await self._check_prank_match(
                request.prank_slug,
                request.prompt
            )
        
        # Step 2: Route to appropriate model (unless prank matched)
        if prank_match:
            # Load pre-generated prank image
            image = Image.open(prank_match["image_path"])
            routing_decision = RoutingDecision(
                is_prank=True,
                prank_id=prank_match["trigger_id"],
                chosen_model_id="prank_cache",
                reason="Matched prank trigger",
                tags=["prank"],
                candidates=[]
            )
            was_prank = True
            matched_trigger_id = prank_match["trigger_id"]
        else:
            # Get routing decision
            if request.engine_override:
                routing_decision = RoutingDecision(
                    is_prank=False,
                    prank_id=None,
                    chosen_model_id=request.engine_override,
                    reason="Manual engine override",
                    tags=["override"],
                    candidates=[]
                )
            else:
                routing_decision = await self._route_prompt(request.prompt)
            
            # Execute generation
            image = await self._execute_generation(
                routing_decision.chosen_model_id,
                request
            )
            was_prank = False
            matched_trigger_id = None
        
        # Step 3: Save everything
        generation_time_ms = int((time.time() - start_time) * 1000)
        
        storage_record = self.storage.save_generation(
            image=image,
            prompt=request.prompt,
            routing_decision=routing_decision.dict(),
            generation_params={
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "width": request.width,
                "height": request.height,
                "seed": request.seed
            },
            prank_info={
                "was_prank": was_prank,
                "prank_slug": request.prank_slug,
                "matched_trigger_id": matched_trigger_id
            } if request.prank_slug else None
        )
        
        # Step 4: Database logging
        db_generation = Generation(
            id=storage_record["id"],
            prompt=request.prompt,
            routing_decision=routing_decision.dict(),
            chosen_model=routing_decision.chosen_model_id,
            model_scores=routing_decision.candidates,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            seed=request.seed,
            status=GenerationStatus.COMPLETED,
            image_path=storage_record["paths"]["full_image"],
            thumbnail_path=storage_record["paths"]["thumbnail"],
            generation_time_ms=generation_time_ms,
            was_prank=was_prank,
            matched_trigger_id=matched_trigger_id,
            session_id=request.session_id,
            created_at=storage_record["timestamp"]
        )
        self.db.add(db_generation)
        self.db.commit()
        
        return GenerationResult(
            image=image,
            generation_id=storage_record["id"],
            routing_decision=routing_decision,
            was_prank=was_prank,
            matched_trigger_id=matched_trigger_id,
            generation_time_ms=generation_time_ms,
            storage_paths=storage_record["paths"],
            thumbnail_blob=base64.b64decode(storage_record["thumbnail_blob"])
        )
    
    async def _route_prompt(self, prompt: str) -> RoutingDecision:
        """
        Get routing decision from LLM with detailed scoring.
        """
        try:
            # Get base routing from LLM
            decision = self.router.route(prompt)
            
            # Enhance with registry metadata
            model_info = self.registry.get_model_info(decision.chosen_model_id)
            if model_info:
                decision.tags.extend(model_info.get("tags", []))
            
            return decision
            
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            # Fallback to FLUX
            return RoutingDecision(
                is_prank=False,
                prank_id=None,
                chosen_model_id="flux_dev",
                reason="Routing error - using default",
                tags=["fallback"],
                candidates=[]
            )
    
    async def _check_prank_match(
        self,
        prank_slug: str,
        prompt: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check if prompt matches any triggers for the prank set.
        Uses similarity matching with configurable thresholds.
        """
        # Get prank set from cache or DB
        if prank_slug not in self._prank_cache:
            prank_set = self.db.query(PrankSet).filter_by(
                share_slug=prank_slug,
                is_active=True
            ).first()
            
            if not prank_set:
                return None
            
            self._prank_cache[prank_slug] = {
                "id": str(prank_set.id),
                "triggers": [
                    {
                        "id": str(t.id),
                        "prompt": t.trigger_prompt,
                        "threshold": t.similarity_threshold,
                        "exact_only": t.exact_match_only,
                        "case_sensitive": t.case_sensitive,
                        "image_path": t.image_versions.get(t.current_version)
                    }
                    for t in prank_set.triggers
                ]
            }
        
        prank_data = self._prank_cache.get(prank_slug)
        if not prank_data:
            return None
        
        # Check triggers in priority order
        for trigger in sorted(
            prank_data["triggers"],
            key=lambda t: t.get("priority", 0),
            reverse=True
        ):
            if self._prompt_matches_trigger(prompt, trigger):
                # Update trigger stats
                self.db.execute(
                    "UPDATE prank_triggers SET trigger_count = trigger_count + 1, "
                    "last_triggered = NOW() WHERE id = :id",
                    {"id": trigger["id"]}
                )
                return {
                    "trigger_id": trigger["id"],
                    "image_path": trigger["image_path"]
                }
        
        return None
    
    def _prompt_matches_trigger(
        self,
        prompt: str,
        trigger: Dict[str, Any]
    ) -> bool:
        """
        Check if a prompt matches a trigger using various strategies.
        """
        trigger_prompt = trigger["prompt"]
        
        # Handle case sensitivity
        if not trigger.get("case_sensitive", False):
            prompt = prompt.lower()
            trigger_prompt = trigger_prompt.lower()
        
        # Exact match check
        if trigger.get("exact_only", False):
            return prompt.strip() == trigger_prompt.strip()
        
        # Similarity matching (simplified - enhance with embeddings)
        # For now, use simple substring and word overlap
        prompt_words = set(prompt.split())
        trigger_words = set(trigger_prompt.split())
        
        if not trigger_words:
            return False
        
        overlap = len(prompt_words & trigger_words)
        similarity = overlap / len(trigger_words)
        
        return similarity >= trigger.get("threshold", 0.85)
    
    async def _execute_generation(
        self,
        model_id: str,
        request: GenerationRequest
    ) -> Image.Image:
        """
        Execute the actual image generation with the chosen model.
        """
        executor = self.model_executors.get(model_id)
        if not executor:
            raise ValueError(f"Unknown model: {model_id}")
        
        # Run generation (could be made async with thread pool)
        image = executor(
            prompt=request.prompt,
            num_steps=request.num_inference_steps,
            guidance=request.guidance_scale,
            width=request.width,
            height=request.height,
            seed=request.seed
        )
        
        return image
    
    def _refresh_prank_cache(self):
        """Periodically refresh prank cache from database"""
        # This could be called by a background task
        self._prank_cache.clear()
```

### Phase 4: FastAPI Endpoints Implementation

```python
# backend/app/api/routes.py
from __future__ import annotations

import secrets
import string
from typing import Optional, List
from fastapi import (
    APIRouter, HTTPException, Depends, File, UploadFile,
    Form, Query, BackgroundTasks
)
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import base64

from ..database import get_db
from ..models.orchestrator import ModelOrchestrator, GenerationRequest
from ..storage.manager import StorageManager
from ..schemas import (
    PrankSetCreate, PrankSetResponse, PrankTriggerCreate,
    GenerationResponse, PrankGenerationRequest
)

router = APIRouter()

# Initialize singletons
storage_manager = StorageManager()
orchestrator = None  # Initialized per request with DB session

def generate_unique_slug(length: int = 6) -> str:
    """Generate a unique URL slug"""
    chars = string.ascii_lowercase + string.digits
    return ''.join(secrets.choice(chars) for _ in range(length))

@router.post("/api/pranks/create")
async def create_prank_set(
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db)
) -> PrankSetResponse:
    """
    Create a new prank set with builder and share URLs.
    No authentication required - just return the secret builder URL.
    """
    # Generate unique slugs
    builder_slug = generate_unique_slug(8)
    share_slug = generate_unique_slug(6)
    
    # Ensure uniqueness
    while db.query(PrankSet).filter(
        (PrankSet.builder_slug == builder_slug) |
        (PrankSet.share_slug == share_slug)
    ).first():
        builder_slug = generate_unique_slug(8)
        share_slug = generate_unique_slug(6)
    
    # Create prank set
    prank_set = PrankSet(
        builder_slug=builder_slug,
        share_slug=share_slug,
        title=title,
        description=description
    )
    db.add(prank_set)
    db.commit()
    
    return PrankSetResponse(
        id=str(prank_set.id),
        builder_url=f"https://promptpics.ai/customize/{builder_slug}",
        share_url=f"https://promptpics.ai/p/{share_slug}",
        title=prank_set.title,
        created_at=prank_set.created_at.isoformat()
    )

@router.post("/api/pranks/{builder_slug}/triggers")
async def add_prank_trigger(
    builder_slug: str,
    trigger_prompt: str = Form(...),
    image: UploadFile = File(...),
    similarity_threshold: float = Form(0.85),
    priority: int = Form(0),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> PrankTriggerResponse:
    """
    Add a trigger to a prank set using the builder slug.
    """
    # Verify builder access
    prank_set = db.query(PrankSet).filter_by(
        builder_slug=builder_slug,
        is_active=True
    ).first()
    
    if not prank_set:
        raise HTTPException(status_code=404, detail="Prank set not found")
    
    # Save image
    image_data = await image.read()
    pil_image = Image.open(io.BytesIO(image_data))
    
    # Create trigger
    trigger = PrankTrigger(
        prank_set_id=prank_set.id,
        trigger_prompt=trigger_prompt,
        similarity_threshold=similarity_threshold,
        priority=priority
    )
    db.add(trigger)
    db.flush()  # Get trigger ID
    
    # Save image versions
    paths = storage_manager.save_prank_image(
        trigger_id=str(trigger.id),
        image=pil_image,
        version_number=1
    )
    
    trigger.image_versions = {"original": paths["original"], "current": paths["web"]}
    trigger.thumbnail_path = paths["thumbnail"]
    trigger.current_version = "original"
    
    # Update prank set stats
    prank_set.total_triggers += 1
    
    db.commit()
    
    # Background: compute prompt embedding
    background_tasks.add_task(
        compute_prompt_embedding,
        trigger.id,
        trigger_prompt
    )
    
    return PrankTriggerResponse(
        id=str(trigger.id),
        trigger_prompt=trigger_prompt,
        thumbnail_url=f"/api/images/thumb/{trigger.id}",
        created_at=trigger.created_at.isoformat()
    )

@router.get("/api/pranks/{builder_slug}")
async def get_prank_set_details(
    builder_slug: str,
    db: Session = Depends(get_db)
) -> PrankSetDetailResponse:
    """
    Get full details of a prank set for the builder view.
    """
    prank_set = db.query(PrankSet).filter_by(
        builder_slug=builder_slug
    ).first()
    
    if not prank_set:
        raise HTTPException(status_code=404, detail="Prank set not found")
    
    triggers = [
        {
            "id": str(t.id),
            "prompt": t.trigger_prompt,
            "thumbnail_url": f"/api/images/thumb/{t.id}",
            "version": t.current_version,
            "trigger_count": t.trigger_count,
            "last_triggered": t.last_triggered.isoformat() if t.last_triggered else None
        }
        for t in prank_set.triggers
    ]
    
    return PrankSetDetailResponse(
        id=str(prank_set.id),
        title=prank_set.title,
        description=prank_set.description,
        builder_url=f"https://promptpics.ai/customize/{builder_slug}",
        share_url=f"https://promptpics.ai/p/{prank_set.share_slug}",
        triggers=triggers,
        total_triggers=prank_set.total_triggers,
        successful_pranks=prank_set.successful_pranks,
        created_at=prank_set.created_at.isoformat(),
        stats={
            "access_count": prank_set.access_count,
            "last_accessed": prank_set.last_accessed.isoformat() if prank_set.last_accessed else None
        }
    )

@router.delete("/api/pranks/{builder_slug}/triggers/{trigger_id}")
async def delete_prank_trigger(
    builder_slug: str,
    trigger_id: str,
    db: Session = Depends(get_db)
):
    """Delete a trigger from a prank set."""
    # Verify builder access and ownership
    trigger = db.query(PrankTrigger).join(PrankSet).filter(
        PrankSet.builder_slug == builder_slug,
        PrankTrigger.id == trigger_id
    ).first()
    
    if not trigger:
        raise HTTPException(status_code=404, detail="Trigger not found")
    
    db.delete(trigger)
    db.commit()
    
    return {"message": "Trigger deleted successfully"}

@router.post("/api/pranks/{builder_slug}/triggers/{trigger_id}/edit")
async def edit_prank_trigger_image(
    builder_slug: str,
    trigger_id: str,
    image: UploadFile = File(...),
    edit_description: str = Form(None),
    db: Session = Depends(get_db)
) -> PrankTriggerEditResponse:
    """
    Upload a new version of a prank image.
    Preserves all previous versions.
    """
    trigger = db.query(PrankTrigger).join(PrankSet).filter(
        PrankSet.builder_slug == builder_slug,
        PrankTrigger.id == trigger_id
    ).first()
    
    if not trigger:
        raise HTTPException(status_code=404, detail="Trigger not found")
    
    # Determine next version number
    current_versions = trigger.image_versions or {}
    version_numbers = [
        int(k.replace("v", ""))
        for k in current_versions.keys()
        if k.startswith("v") and k[1:].isdigit()
    ]
    next_version = max(version_numbers, default=0) + 1
    
    # Save new image version
    image_data = await image.read()
    pil_image = Image.open(io.BytesIO(image_data))
    
    paths = storage_manager.save_prank_image(
        trigger_id=str(trigger.id),
        image=pil_image,
        version_number=next_version,
        edit_description=edit_description
    )
    
    # Update trigger with new version
    trigger.image_versions[f"v{next_version}"] = paths["original"]
    trigger.current_version = f"v{next_version}"
    trigger.thumbnail_path = paths["thumbnail"]
    
    db.commit()
    
    return PrankTriggerEditResponse(
        version_number=next_version,
        paths=paths,
        edit_description=edit_description
    )

@router.post("/api/generate")
async def generate_image(
    request: GenerationRequest,
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> GenerationResponse:
    """
    Main generation endpoint with full routing and prank support.
    """
    # Initialize orchestrator with DB session
    orch = ModelOrchestrator(storage_manager, db)
    
    # Execute generation
    result = await orch.generate(request)
    
    # Background: update statistics
    if request.prank_slug and result.was_prank:
        background_tasks.add_task(
            update_prank_stats,
            request.prank_slug,
            result.matched_trigger_id
        )
    
    return GenerationResponse(
        image_base64=base64.b64encode(
            result.image.tobytes()
        ).decode('utf-8'),
        generation_id=result.generation_id,
        model_used=result.routing_decision.chosen_model_id,
        was_prank=result.was_prank,
        routing_details=result.routing_decision.dict(),
        generation_time_ms=result.generation_time_ms,
        thumbnail_base64=base64.b64encode(
            result.thumbnail_blob
        ).decode('utf-8')
    )

@router.get("/api/generations/logs")
async def get_generation_logs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session_id: Optional[str] = Query(None),
    db: Session = Depends(get_db)
) -> List[GenerationLogResponse]:
    """
    Get generation logs for analytics and debugging.
    """
    query = db.query(Generation).order_by(
        Generation.created_at.desc()
    )
    
    if session_id:
        query = query.filter(Generation.session_id == session_id)
    
    generations = query.offset(offset).limit(limit).all()
    
    return [
        GenerationLogResponse(
            id=str(g.id),
            prompt=g.prompt,
            model_used=g.chosen_model,
            was_prank=g.was_prank,
            created_at=g.created_at.isoformat(),
            generation_time_ms=g.generation_time_ms,
            thumbnail_url=f"/api/images/thumb/{g.id}",
            routing_decision=g.routing_decision
        )
        for g in generations
    ]

@router.get("/api/images/thumb/{image_id}")
async def get_thumbnail(
    image_id: str,
    db: Session = Depends(get_db)
):
    """Serve thumbnail images."""
    # Check if it's a generation or trigger ID
    generation = db.query(Generation).filter_by(id=image_id).first()
    if generation and generation.thumbnail_path:
        path = Path(generation.thumbnail_path)
        if path.exists():
            return StreamingResponse(
                io.BytesIO(path.read_bytes()),
                media_type="image/webp"
            )
    
    trigger = db.query(PrankTrigger).filter_by(id=image_id).first()
    if trigger and trigger.thumbnail_path:
        path = Path(trigger.thumbnail_path)
        if path.exists():
            return StreamingResponse(
                io.BytesIO(path.read_bytes()),
                media_type="image/webp"
            )
    
    raise HTTPException(status_code=404, detail="Thumbnail not found")

# Background tasks
async def compute_prompt_embedding(trigger_id: str, prompt: str):
    """Compute and store prompt embeddings for similarity matching."""
    # Implementation using sentence transformers or similar
    pass

async def update_prank_stats(prank_slug: str, trigger_id: str):
    """Update prank and trigger statistics."""
    # Implementation to update counters
    pass
```

### Phase 5: Database Initialization and Migration

```python
# backend/app/database/__init__.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import os
from .models import Base

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./data/promptpics.db"
)

# For production, use PostgreSQL:
# DATABASE_URL = "postgresql://user:password@localhost/promptpics"

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
else:
    engine = create_engine(
        DATABASE_URL,
        pool_size=20,
        max_overflow=40,
        pool_pre_ping=True,
        echo=False
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

def get_db() -> Session:
    """Dependency to get DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Phase 6: Complete Pydantic Schemas

```python
# backend/app/schemas.py
from __future__ import annotations

from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field, validator
import uuid

# Generation Schemas
class GenerationRequest(BaseModel):
    """Main generation request"""
    prompt: str = Field(..., description="Text prompt for image generation")
    prank_slug: Optional[str] = Field(None, description="Prank share slug if applicable")
    num_inference_steps: int = Field(28, ge=10, le=60)
    guidance_scale: float = Field(4.0, ge=0.0, le=20.0)
    width: int = Field(1024, ge=512, le=1536)
    height: int = Field(1024, ge=512, le=1536)
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    engine: Literal["auto", "flux_dev", "sd3_medium", "realvis_xl", "logo_sdxl"] = Field(
        "auto",
        description="Model selection strategy"
    )
    session_id: Optional[str] = Field(None, description="Browser session for tracking")

class ModelScore(BaseModel):
    """Individual model scoring"""
    model_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str

class RoutingDecision(BaseModel):
    """Complete routing decision from LLM"""
    is_prank: bool = False
    prank_id: Optional[str] = None
    chosen_model_id: str
    reason: str
    tags: List[str] = Field(default_factory=list)
    candidates: List[ModelScore] = Field(default_factory=list)
    confidence_score: Optional[float] = None

class GenerationResponse(BaseModel):
    """Generation result"""
    image_base64: str
    generation_id: str
    model_used: str
    was_prank: bool
    routing_details: RoutingDecision
    generation_time_ms: int
    thumbnail_base64: str

# Prank Schemas
class PrankSetCreate(BaseModel):
    """Create a new prank set"""
    title: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None

class PrankSetResponse(BaseModel):
    """Prank set creation response"""
    id: str
    builder_url: str  # Secret URL for managing
    share_url: str    # Public URL for sharing
    title: Optional[str]
    created_at: str

class PrankTriggerCreate(BaseModel):
    """Add a trigger to prank set"""
    trigger_prompt: str
    similarity_threshold: float = Field(0.85, ge=0.0, le=1.0)
    priority: int = Field(0, ge=0, le=100)
    exact_match_only: bool = False
    case_sensitive: bool = False

class PrankTriggerResponse(BaseModel):
    """Trigger creation response"""
    id: str
    trigger_prompt: str
    thumbnail_url: str
    created_at: str
    version_number: int = 1

class PrankTriggerEditResponse(BaseModel):
    """Image edit response"""
    version_number: int
    paths: Dict[str, str]
    edit_description: Optional[str]

class PrankSetDetailResponse(BaseModel):
    """Full prank set details"""
    id: str
    title: Optional[str]
    description: Optional[str]
    builder_url: str
    share_url: str
    triggers: List[Dict[str, Any]]
    total_triggers: int
    successful_pranks: int
    created_at: str
    stats: Dict[str, Any]

# Logging Schemas
class GenerationLogResponse(BaseModel):
    """Generation log entry"""
    id: str
    prompt: str
    model_used: str
    was_prank: bool
    created_at: str
    generation_time_ms: Optional[int]
    thumbnail_url: str
    routing_decision: Dict[str, Any]

class RoutingLogEntry(BaseModel):
    """Routing decision for training"""
    generation_id: str
    prompt: str
    router_model_used: str
    chosen_model: str
    model_candidates: List[ModelScore]
    confidence_score: Optional[float]
    routing_time_ms: int
    human_verified: bool = False
    correct_choice: Optional[bool] = None
    preferred_model: Optional[str] = None
```

### Phase 7: Main Application Entry Point

```python
# backend/app/main.py
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .database import init_db
from .api import routes
from .storage.manager import StorageManager
from .models.model_registry import ModelRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize app resources on startup"""
    logger.info("Initializing PromptPics backend...")
    
    # Initialize database
    init_db()
    logger.info("Database initialized")
    
    # Initialize storage
    storage = StorageManager()
    logger.info(f"Storage initialized at {storage.base_path}")
    
    # Pre-load model registry
    registry = ModelRegistry()
    registry.initialize()
    logger.info("Model registry loaded")
    
    yield
    
    # Cleanup
    logger.info("Shutting down PromptPics backend...")

# Create FastAPI app
app = FastAPI(
    title="PromptPics Multi-Model Backend",
    version="2.0.0",
    description="Advanced image generation with model routing and prank system",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://promptpics.ai",
        "https://www.promptpics.ai",
        "https://*.replit.app",
        "https://*.repl.co",
        "http://localhost:3000",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Generation-ID", "X-Model-Used"]
)

# Mount routes
app.include_router(routes.router)

# Serve static files for images (optional)
app.mount("/static", StaticFiles(directory="data"), name="static")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "models_available": [
            "flux_dev",
            "sd3_medium",
            "realvis_xl",
            "logo_sdxl"
        ]
    }
```

### Phase 8: Deployment Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "7999:7999"
    volumes:
      - ./data:/data
      - ./models:/models
    environment:
      - DATABASE_URL=postgresql://promptpics:password@db:5432/promptpics
      - FLUX_TEXT_MODEL_ID=black-forest-labs/FLUX.1-dev
      - SD3_MODEL_ID=stabilityai/stable-diffusion-3-medium-diffusers
      - REALVIS_MODEL_ID=SG161222/RealVisXL_V4.0
      - LOGO_MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0
      - ROUTER_LLM_MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct
      - HF_TOKEN=${HF_TOKEN}
    depends_on:
      - db
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=promptpics
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=promptpics
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
      
  ngrok:
    image: ngrok/ngrok:latest
    command: 
      - "start"
      - "--all"
      - "--config"
      - "/etc/ngrok.yml"
    volumes:
      - ./ngrok.yml:/etc/ngrok.yml
    ports:
      - "4040:4040"
    depends_on:
      - backend

volumes:
  postgres_data:
```

### Installation Instructions for Your Agent

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Initialize database
python -m app.database init

# 3. Create storage directories
python -c "from app.storage.manager import StorageManager; StorageManager()"

# 4. Start the backend
uvicorn app.main:app --host 0.0.0.0 --port 7999 --reload

# 5. Start ngrok (in another terminal)
ngrok start --all --config ~/.config/ngrok/ngrok.yml
```

### Requirements.txt

```txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.0
sqlalchemy==2.0.25
psycopg2-binary==2.9.9
alembic==1.13.1
pillow==10.2.0
torch==2.1.2
transformers==4.36.2
diffusers==0.25.0
accelerate==0.25.0
safetensors==0.4.1
sentencepiece==0.1.99
sentence-transformers==2.2.2
numpy==1.24.4
scipy==1.11.4
redis==5.0.1
httpx==0.25.2
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
```

This comprehensive implementation provides:

1. **Complete database schema** with versioning, tracking, and analytics
2. **Storage management** with thumbnails and full images
3. **Multi-model orchestration** with intelligent routing
4. **Prank system** with similarity matching
5. **No authentication required** - just unique URLs
6. **Full logging** for training and analytics
7. **RESTful API** for all operations
8. **Production-ready architecture** with proper error handling

The system tracks everything you requested:
- All prompts and images
- Routing decisions for training
- Multiple image versions
- Thumbnails for fast loading
- Complete generation history
- Prank usage statistics

Your coding agent can implement this incrementally, starting with the database and storage layers, then adding the routing system, and finally the prank features.