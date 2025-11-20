"""Pydantic models shared across endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TextGenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for FLUX generation")
    num_inference_steps: int = Field(4, ge=1, le=50)
    guidance_scale: float = Field(0.0, ge=0.0, le=50.0)
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    seed: int | None = Field(None, description="Optional RNG seed for reproducibility")


class ImageResponse(BaseModel):
    image_base64: str
