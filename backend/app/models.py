"""SQLAlchemy models for pranks and triggers."""

from __future__ import annotations

import uuid

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, Index, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class Prank(Base):
    __tablename__ = "pranks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Legacy slug kept for compatibility; share_slug is the canonical public slug.
    slug = Column(String(16), unique=True, nullable=True, index=True)
    share_slug = Column(String(16), unique=True, nullable=False, index=True)
    builder_slug = Column(String(16), unique=True, nullable=False, index=True)
    title = Column(Text, nullable=True)
    session_id = Column(String(100), nullable=True)
    view_count = Column(Integer, default=0)
    is_vip = Column(Boolean, default=False, nullable=False)  # VIP pranks (like "imagine") get special treatment
    is_admin_only = Column(Boolean, default=False, nullable=False)  # Only admin can edit this prank
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    triggers = relationship(
        "PrankTrigger",
        back_populates="prank",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class PrankTrigger(Base):
    __tablename__ = "prank_triggers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prank_id = Column(UUID(as_uuid=True), ForeignKey("pranks.id", ondelete="CASCADE"), nullable=False, index=True)
    trigger_text = Column(Text, nullable=False)
    image_path = Column(Text, nullable=False)
    thumbnail_path = Column(Text, nullable=True)
    match_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    prank = relationship("Prank", back_populates="triggers")


class GenerationLog(Base):
    """
    Minimal generation log to persist outputs and routing metadata.
    """

    __tablename__ = "generation_logs"
    __table_args__ = (
        Index("idx_generation_session", "session_id"),
        Index("idx_generation_created", "created_at"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prompt = Column(Text, nullable=False)
    model_id = Column(String(50), nullable=False)
    router_json = Column(Text, nullable=True)
    image_path = Column(Text, nullable=False)
    thumbnail_path = Column(Text, nullable=False)
    prank_id = Column(UUID(as_uuid=True), ForeignKey("pranks.id"), nullable=True)
    share_slug = Column(String(16), nullable=True)
    session_id = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class GenerationMetric(Base):
    """
    Detailed timing and parameter capture for latency distribution studies.
    Separate from GenerationLog to avoid changing existing consumer logic.
    """

    __tablename__ = "generation_metrics"
    __table_args__ = (
        Index("idx_metrics_model_time", "model_used", "started_at"),
        Index("idx_metrics_started", "started_at"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prompt = Column(Text, nullable=False)
    prompt_length = Column(Integer, nullable=False)
    model_used = Column(String(50), nullable=False)
    engine_requested = Column(String(50), nullable=True)
    num_inference_steps = Column(Integer, nullable=True)
    guidance_scale = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    seed = Column(Integer, nullable=True)
    tf32_enabled = Column(Boolean, nullable=True)
    is_synthetic = Column(Boolean, nullable=False, default=False)
    is_prank = Column(Boolean, nullable=False, default=False)
    queue_position_at_start = Column(Integer, nullable=True)
    queue_wait_ms = Column(Integer, nullable=True)
    duration_ms = Column(Integer, nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    ended_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    router_json = Column(Text, nullable=True)
    session_id = Column(String(100), nullable=True)
    share_slug = Column(String(16), nullable=True)
    prompt_metadata = Column(Text, nullable=True)  # e.g., benchmark ids/categories


class BenchRun(Base):
    __tablename__ = "bench_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(100), nullable=True)
    prompt = Column(Text, nullable=False)
    prompt_length = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False, default="queued")  # queued, running, done, error
    current_engine = Column(String(50), nullable=True)
    resolution = Column(Integer, nullable=True)  # e.g., 512 or 1024
    tf32_enabled = Column(Boolean, nullable=True)
    engines_json = Column(Text, nullable=True)  # original request engines config
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    results = relationship(
        "BenchRunResult",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class BenchRunResult(Base):
    __tablename__ = "bench_run_results"
    __table_args__ = (
        Index("idx_bench_results_run", "run_id"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("bench_runs.id", ondelete="CASCADE"), nullable=False)
    engine = Column(String(50), nullable=False)
    steps = Column(Integer, nullable=False)
    guidance = Column(Float, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    seed = Column(Integer, nullable=False)
    elapsed_ms = Column(Integer, nullable=False)
    tf32_enabled = Column(Boolean, nullable=False, default=True)
    image_path = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Extended metrics for data science
    model_load_ms = Column(Integer, nullable=True)  # Time to load model into GPU
    warmup_ms = Column(Integer, nullable=True)  # Time for warmup inference
    text_encoder_ms = Column(Integer, nullable=True)  # Remote text encoder latency (FLUX2)
    inference_only_ms = Column(Integer, nullable=True)  # Pure inference time (no text encoding)

    # GPU memory metrics (in MB for easier reading)
    gpu_mem_before_load_mb = Column(Integer, nullable=True)  # GPU memory before model load
    gpu_mem_after_load_mb = Column(Integer, nullable=True)  # GPU memory after model load
    gpu_mem_peak_mb = Column(Integer, nullable=True)  # Peak GPU memory during inference
    gpu_mem_after_inference_mb = Column(Integer, nullable=True)  # GPU memory after inference
    gpu_mem_allocated_mb = Column(Integer, nullable=True)  # torch.cuda.memory_allocated
    gpu_mem_reserved_mb = Column(Integer, nullable=True)  # torch.cuda.memory_reserved

    # Hardware info
    gpu_name = Column(String(100), nullable=True)  # e.g., "NVIDIA GeForce RTX 5090"
    gpu_compute_capability = Column(String(10), nullable=True)  # e.g., "9.0"
    gpu_total_mem_mb = Column(Integer, nullable=True)  # Total GPU memory

    # Software versions for reproducibility
    torch_version = Column(String(50), nullable=True)
    diffusers_version = Column(String(50), nullable=True)
    python_version = Column(String(20), nullable=True)

    # Model metadata
    model_dtype = Column(String(20), nullable=True)  # e.g., "bfloat16", "float16"
    quantization = Column(String(20), nullable=True)  # e.g., "4bit", "8bit", "none"
    uses_remote_encoder = Column(Boolean, nullable=True)  # True for FLUX2

    run = relationship("BenchRun", back_populates="results")
