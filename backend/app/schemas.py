"""Pydantic models shared across endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class TextGenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    prompt: str = Field(..., description="Text prompt for FLUX generation")
    num_inference_steps: int = Field(4, ge=1, le=50)
    guidance_scale: float = Field(0.0, ge=0.0, le=50.0)
    width: int = Field(768, ge=256, le=2048)
    height: int = Field(768, ge=256, le=2048)
    seed: int | None = Field(None, description="Optional RNG seed for reproducibility")
    engine: Literal["auto", "flux_dev", "realvis_xl", "sd3_medium", "hidream_dev"] = Field(
        "auto", description="Engine override; 'auto' uses router"
    )
    session_id: str | None = Field(None, alias="sessionId", description="Optional session identifier")
    benchmark_meta: dict | None = Field(None, alias="benchmarkMeta", description="Optional benchmark metadata blob")

class RoutingMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    chosen_model_id: str
    scores: dict[str, float] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    reason: str = ""


class MatchedTrigger(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str
    trigger_text: str = Field(..., alias="trigger_text", serialization_alias="triggerText")
    image_base64: str = Field(..., alias="image_base64", serialization_alias="imageBase64")
    thumbnail_base64: str | None = Field(
        None, alias="thumbnail_base64", serialization_alias="thumbnailBase64"
    )


class ImageResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)
    image_base64: str
    generation_id: str | None = None
    model_id: str | None = None
    model_used: str | None = None
    image_url: str | None = Field(None, alias="imageUrl")
    prompt: str | None = None
    processing_time_ms: int | None = Field(None, alias="processingTimeMs")
    queue_wait_ms: int | None = Field(None, alias="queueWaitMs")
    distribution: list[int] | None = None
    distribution_all: list[int] | None = Field(None, alias="distributionAll")
    was_prank: bool | None = Field(None, alias="was_prank", serialization_alias="wasPrank")
    matched_trigger_id: str | None = Field(None, alias="matched_trigger_id", serialization_alias="matched_trigger_id")
    matched_trigger_text: str | None = Field(
        None, alias="matched_trigger_text", serialization_alias="matchedTriggerText"
    )
    generation_time_ms: int | None = None
    thumbnail_base64: str | None = None
    image_path: str | None = None
    thumbnail_path: str | None = None
    router_metadata: RoutingMetadata | None = None
    is_prank_match: bool | None = Field(None, alias="is_prank_match", serialization_alias="isPrankMatch")
    matched_trigger: MatchedTrigger | None = None


class PrankMetadataCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    title: str | None = Field(None, description="Optional title for your prank link")
    session_id: str | None = Field(None, alias="sessionId", description="Optional session identifier")

class PrankCreateResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)
    id: str
    slug: str  # share slug (legacy field for backward compatibility)
    share_slug: str = Field(..., alias="shareSlug")  # Explicit shareSlug field
    builder_slug: str = Field(..., alias="builderSlug")
    title: str | None
    session_id: str | None = Field(None, alias="sessionId")
    share_url: str = Field(..., alias="shareUrl")
    builder_url: str = Field(..., alias="builderUrl")
    created_at: str = Field(..., alias="createdAt")
    view_count: int = Field(..., alias="viewCount")
    is_vip: bool = Field(False, alias="isVip")
    triggers: list["PrankTriggerInfo"] = []

class PrankTriggerCreateRequest(BaseModel):
    trigger_text: str = Field(..., description="Prompt that should trigger the prank image")
    image_relative_path: str = Field(
        ..., description="Relative path under PRANK_IMAGE_ROOT/<slug>/ pointing to an existing prank image"
    )


class PrankTriggerCreateResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)
    id: str
    trigger_text: str = Field(..., alias="triggerText")
    image_path: str = Field(..., alias="imagePath")
    thumbnail_path: str | None = Field(None, alias="thumbnailPath")


class PrankGenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    prompt: str = Field(..., description="User prompt on a prank page")
    num_inference_steps: int | None = Field(None, ge=1, le=50)
    guidance_scale: float | None = Field(None, ge=0.0, le=50.0)
    width: int | None = Field(None, ge=256, le=2048)
    height: int | None = Field(None, ge=256, le=2048)
    seed: int | None = Field(None, description="Optional RNG seed for prank generation")
    session_id: str | None = Field(None, alias="sessionId", description="Optional session identifier")
    engine: Literal["auto", "flux_dev", "realvis_xl", "sd3_medium", "hidream_dev"] | None = Field(
        None, alias="engine", description="Engine override; defaults to auto"
    )

class PrankTriggerUpdateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    trigger_text: str = Field(..., alias="triggerText", description="Updated trigger text")


class PrankTriggerInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)
    id: str
    trigger_text: str = Field(..., alias="triggerText")
    image_base64: str = Field(..., alias="imageBase64")
    thumbnail_base64: str | None = Field(None, alias="thumbnailBase64")
    created_at: str = Field(..., alias="createdAt")
    match_count: int = Field(0, alias="matchCount")

class PrankDetailResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)
    id: str
    slug: str  # legacy field for backward compatibility
    share_slug: str = Field(..., alias="shareSlug")  # Explicit shareSlug field
    builder_slug: str = Field(..., alias="builderSlug")
    title: str | None
    session_id: str | None = Field(None, alias="sessionId")
    share_url: str = Field(..., alias="shareUrl")
    builder_url: str = Field(..., alias="builderUrl")
    created_at: str = Field(..., alias="createdAt")
    view_count: int = Field(..., alias="viewCount")
    is_vip: bool = Field(False, alias="isVip")
    triggers: list[PrankTriggerInfo]


class PrankSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)
    id: str
    slug: str  # legacy field for backward compatibility
    share_slug: str = Field(..., alias="shareSlug")  # Explicit shareSlug field
    builder_slug: str = Field(..., alias="builderSlug")
    title: str | None
    session_id: str | None = Field(None, alias="sessionId")
    share_url: str = Field(..., alias="shareUrl")
    builder_url: str = Field(..., alias="builderUrl")
    created_at: str = Field(..., alias="createdAt")
    view_count: int = Field(..., alias="viewCount")
    trigger_count: int = Field(..., alias="triggerCount")
    is_vip: bool = Field(False, alias="isVip")


class AdminLoginRequest(BaseModel):
    password: str


class AdminLoginResponse(BaseModel):
    admin_token: str


class GenerationLogEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)
    id: str
    prompt: str
    model_id: str
    image_path: str = Field(..., alias="image_path", serialization_alias="imagePath")
    thumbnail_path: str = Field(..., alias="thumbnail_path", serialization_alias="thumbnailPath")
    created_at: str = Field(..., alias="created_at", serialization_alias="createdAt")
    share_slug: str | None = Field(None, alias="share_slug", serialization_alias="shareSlug")
    router_json: dict | None = Field(None, alias="router_json", serialization_alias="routerJson")
    session_id: str | None = Field(None, alias="sessionId", serialization_alias="sessionId")
