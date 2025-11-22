"""SQLAlchemy models for pranks and triggers."""

from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
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
