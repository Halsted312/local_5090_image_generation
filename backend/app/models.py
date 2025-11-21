"""SQLAlchemy models for pranks and triggers."""

from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class Prank(Base):
    __tablename__ = "pranks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    slug = Column(String(16), unique=True, nullable=False, index=True)
    title = Column(Text, nullable=True)
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
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    prank = relationship("Prank", back_populates="triggers")
