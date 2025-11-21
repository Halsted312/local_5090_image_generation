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


class PrankMetadataCreate(BaseModel):
    title: str | None = Field(None, description="Optional title for your prank link")
    slug: str | None = Field(None, description="Optional custom slug (3-16 chars). Leave empty to auto-generate.")


class PrankCreateResponse(BaseModel):
    prank_id: str
    slug: str
    share_url: str


class PrankTriggerCreateRequest(BaseModel):
    trigger_text: str = Field(..., description="Prompt that should trigger the prank image")
    image_relative_path: str = Field(
        ..., description="Relative path under PRANK_IMAGE_ROOT/<slug>/ pointing to an existing prank image"
    )


class PrankTriggerCreateResponse(BaseModel):
    id: str
    trigger_text: str
    image_path: str


class PrankGenerateRequest(BaseModel):
    prompt: str = Field(..., description="User prompt on a prank page")
    num_inference_steps: int | None = Field(None, ge=1, le=50)
    guidance_scale: float | None = Field(None, ge=0.0, le=50.0)
    width: int | None = Field(None, ge=256, le=2048)
    height: int | None = Field(None, ge=256, le=2048)
    seed: int | None = Field(None, description="Optional RNG seed for prank generation")


class PrankTriggerInfo(BaseModel):
    id: str
    trigger_text: str
    image_base64: str


class PrankDetailResponse(BaseModel):
    prank_id: str
    slug: str
    title: str | None
    triggers: list[PrankTriggerInfo]
