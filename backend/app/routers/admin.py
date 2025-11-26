"""Admin authentication and management endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Header, Request
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Prank, PrankTrigger
from ..schemas import (
    AdminLoginRequest,
    AdminLoginResponse,
    PrankCreateResponse,
    PrankSummary,
)
from ..services.deps import (
    ADMIN_LOGIN_MAX_ATTEMPTS,
    ADMIN_LOGIN_WINDOW,
    ADMIN_PASSWORD,
    ADMIN_TOKEN_TTL_SECONDS,
    client_ip,
    generate_unique_slug,
    get_admin_token,
    get_admin_token_issued,
    issue_admin_token,
)
from ..storage import load_prank_image_base64

router = APIRouter(prefix="/api/admin", tags=["admin"])

# In-memory rate limit store (will be set from main app state)
_admin_login_attempts: dict = {}


def _check_admin_rate_limit(request: Request) -> None:
    """
    Simple in-memory rate limiter per client IP for admin login.
    """
    ip = client_ip(request)
    now = time.time()
    window = ADMIN_LOGIN_WINDOW
    limit = ADMIN_LOGIN_MAX_ATTEMPTS
    history = [t for t in _admin_login_attempts.get(ip, []) if now - t < window]
    if len(history) >= limit:
        raise HTTPException(status_code=429, detail="Too many admin login attempts; try again later")
    history.append(now)
    _admin_login_attempts[ip] = history


def require_admin(x_admin_token: str = Header(..., alias="X-Admin-Token")) -> None:
    # Expire tokens after TTL; force re-login.
    if (time.time() - get_admin_token_issued()) > ADMIN_TOKEN_TTL_SECONDS:
        raise HTTPException(status_code=401, detail="Admin token expired; please login again")
    if x_admin_token != get_admin_token():
        raise HTTPException(status_code=403, detail="Admins only")


def _prank_to_response(prank: Prank, triggers: list) -> dict:
    import os
    base_url = os.getenv("FRONTEND_BASE_URL", "https://promptpics.ai")
    share_slug = prank.share_slug
    builder_slug = prank.builder_slug

    # Special URL for VIP prank
    if share_slug == "imagine":
        share_url = f"{base_url}/imagine"
    else:
        share_url = f"{base_url}/p/{share_slug}"

    return {
        "id": str(prank.id),
        "slug": share_slug,  # legacy field
        "shareSlug": share_slug,  # explicit shareSlug field
        "builderSlug": builder_slug,
        "title": prank.title,
        "sessionId": prank.session_id,
        "shareUrl": share_url,
        "builderUrl": f"{base_url}/customize/{builder_slug}",
        "createdAt": prank.created_at.isoformat() if prank.created_at else None,
        "viewCount": prank.view_count or 0,
        "isVip": prank.is_vip if hasattr(prank, 'is_vip') else False,
        "triggers": [
            {
                "id": str(t.id),
                "triggerText": t.trigger_text,
                "imageBase64": load_prank_image_base64(t.image_path),
                "thumbnailBase64": load_prank_image_base64(t.thumbnail_path) if t.thumbnail_path else None,
                "createdAt": t.created_at.isoformat() if t.created_at else None,
                "matchCount": t.match_count or 0,
            }
            for t in triggers
        ],
    }


def _prank_to_summary(prank: Prank, trigger_count: int) -> dict:
    import os
    base_url = os.getenv("FRONTEND_BASE_URL", "https://promptpics.ai")
    share_slug = prank.share_slug
    builder_slug = prank.builder_slug

    # Special URL for VIP prank
    if share_slug == "imagine":
        share_url = f"{base_url}/imagine"
    else:
        share_url = f"{base_url}/p/{share_slug}"

    return {
        "id": str(prank.id),
        "slug": share_slug,  # legacy field
        "shareSlug": share_slug,  # explicit shareSlug field
        "builderSlug": builder_slug,
        "title": prank.title,
        "sessionId": prank.session_id,
        "shareUrl": share_url,
        "builderUrl": f"{base_url}/customize/{builder_slug}",
        "createdAt": prank.created_at.isoformat() if prank.created_at else None,
        "viewCount": prank.view_count or 0,
        "triggerCount": trigger_count,
        "isVip": prank.is_vip if hasattr(prank, 'is_vip') else False,
    }


@router.post("/login", response_model=AdminLoginResponse)
def admin_login(payload: AdminLoginRequest, request: Request) -> AdminLoginResponse:
    # rate-limit attempts per client
    _check_admin_rate_limit(request)
    if not ADMIN_PASSWORD:
        raise HTTPException(status_code=500, detail="Admin password not configured")
    if payload.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid admin password")
    token = issue_admin_token()
    return AdminLoginResponse(admin_token=token)


@router.post("/verify")
def admin_verify(_: None = Depends(require_admin)) -> dict:
    """Simple admin token verification endpoint."""
    return {"status": "ok"}


@router.post("/pranks/vip", response_model=PrankCreateResponse)
def get_or_create_vip_prank(
    _: None = Depends(require_admin),
    db: Session = Depends(get_db)
) -> PrankCreateResponse:
    """
    Idempotent endpoint that ensures the VIP 'imagine' prank exists and returns its details.
    Only admin can call this endpoint.
    """
    # Check if the VIP prank already exists
    existing_prank = db.query(Prank).filter(Prank.share_slug == "imagine").first()

    if existing_prank:
        # Return existing VIP prank
        triggers = list(existing_prank.triggers)
        payload = _prank_to_response(existing_prank, triggers)
        return PrankCreateResponse(**payload)

    # Create the VIP prank
    builder_slug = generate_unique_slug(db, length=8)
    vip_prank = Prank(
        share_slug="imagine",
        builder_slug=builder_slug,
        slug="imagine",  # legacy compatibility
        title="CEO VIP Prank Page",
        session_id=None,  # No session owner, admin-only
        is_vip=True,
        is_admin_only=True,
        view_count=0
    )

    db.add(vip_prank)
    db.commit()
    db.refresh(vip_prank)

    payload = _prank_to_response(vip_prank, [])
    return PrankCreateResponse(**payload)


@router.get("/pranks", response_model=list[PrankSummary])
def admin_list_pranks(
    _: None = Depends(require_admin),
    db: Session = Depends(get_db),
) -> list[PrankSummary]:
    """
    Admin-only list of all pranks (summary only, no base64 payloads).
    """
    pranks = (
        db.query(Prank)
        .order_by(Prank.created_at.desc())
        .all()
    )

    results: list[PrankSummary] = []
    for prank in pranks:
        trigger_count = (
            db.query(PrankTrigger)
            .filter(PrankTrigger.prank_id == prank.id)
            .count()
        )
        payload = _prank_to_summary(prank, trigger_count)
        results.append(PrankSummary(**payload))
    return results
