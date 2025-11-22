"""Pydantic models shared across endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class TextGenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    prompt: str = Field(..., description="Text prompt for FLUX generation")
    num_inference_steps: int = Field(4, ge=1, le=50)
    guidance_scale: float = Field(0.0, ge=0.0, le=50.0)
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    seed: int | None = Field(None, description="Optional RNG seed for reproducibility")
    engine: Literal["auto", "flux_dev", "realvis_xl", "sd3_medium", "logo_sdxl"] = Field(
        "auto", description="Engine override; 'auto' uses router"
    )
    session_id: str | None = Field(None, alias="sessionId", description="Optional session identifier")

class RoutingMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    chosen_model_id: str
    scores: dict[str, float] | None = None
    tags: list[str] | None = None
    reason: str | None = None


class ImageResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    image_base64: str
    generation_id: str | None = None
    model_id: str | None = None
    thumbnail_base64: str | None = None
    image_path: str | None = None
    thumbnail_path: str | None = None
    router_metadata: RoutingMetadata | None = None


class PrankMetadataCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    title: str | None = Field(None, description="Optional title for your prank link")
    session_id: str | None = Field(None, alias="sessionId", description="Optional session identifier")

class PrankCreateResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str
    slug: str  # share slug
    builder_slug: str = Field(..., alias="builderSlug")
    title: str | None
    session_id: str | None = Field(None, alias="sessionId")
    share_url: str = Field(..., alias="shareUrl")
    builder_url: str = Field(..., alias="builderUrl")
    created_at: str = Field(..., alias="createdAt")
    view_count: int = Field(..., alias="viewCount")
    triggers: list["PrankTriggerInfo"] = []

class PrankTriggerCreateRequest(BaseModel):
    trigger_text: str = Field(..., description="Prompt that should trigger the prank image")
    image_relative_path: str = Field(
        ..., description="Relative path under PRANK_IMAGE_ROOT/<slug>/ pointing to an existing prank image"
    )


class PrankTriggerCreateResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str
    trigger_text: str
    image_path: str
    thumbnail_path: str | None = None


class PrankGenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    prompt: str = Field(..., description="User prompt on a prank page")
    num_inference_steps: int | None = Field(None, ge=1, le=50)
    guidance_scale: float | None = Field(None, ge=0.0, le=50.0)
    width: int | None = Field(None, ge=256, le=2048)
    height: int | None = Field(None, ge=256, le=2048)
    seed: int | None = Field(None, description="Optional RNG seed for prank generation")
    session_id: str | None = Field(None, alias="sessionId", description="Optional session identifier")

class PrankTriggerUpdateRequest(BaseModel):
    trigger_text: str = Field(..., description="Updated trigger text")


class PrankTriggerInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str
    trigger_text: str = Field(..., alias="triggerText")
    image_base64: str = Field(..., alias="imageBase64")
    thumbnail_base64: str | None = Field(None, alias="thumbnailBase64")
    created_at: str = Field(..., alias="createdAt")
    match_count: int = Field(0, alias="matchCount")

class PrankDetailResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str
    slug: str
    builder_slug: str = Field(..., alias="builderSlug")
    title: str | None
    session_id: str | None = Field(None, alias="sessionId")
    share_url: str = Field(..., alias="shareUrl")
    builder_url: str = Field(..., alias="builderUrl")
    created_at: str = Field(..., alias="createdAt")
    view_count: int = Field(..., alias="viewCount")
    triggers: list[PrankTriggerInfo]

class GenerationLogEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str
    prompt: str
    model_id: str
    image_path: str
    thumbnail_path: str
    created_at: str
    share_slug: str | None = None
    router_json: dict | None = None
    session_id: str | None = Field(None, alias="sessionId")
