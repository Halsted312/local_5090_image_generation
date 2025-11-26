"""
FLUX Image API + Prank Mode

This is the main FastAPI application entry point.
All route handlers have been modularized into routers/ for maintainability.
"""

from __future__ import annotations

import logging
import threading
from typing import Iterable

import torch
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from .database import Base, engine
from .prank_matching import get_prank_matcher_llm
from .routers import admin, bench, generate, health, pranks
from .services.deps import client_ip
from .services.generation import set_generation_lock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FLUX Image API + Prank Mode")

# Enable TF32 for faster matmul on supported GPUs (e.g., 5090)
torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]

ALLOWED_ORIGINS: Iterable[str] = (
    "https://promptpics.ai",
    "https://www.promptpics.ai",
    "https://app.promptpics.ai",
    "https://app.promptpics.ai",
    "https://promptpics.replit.app",
    "http://localhost:5173",
    "http://localhost:3000",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),  # Only explicit origins, no wildcards
    # Allow Replit preview domains and any subdomain of promptpics.ai (including bare and www).
    allow_origin_regex=r"https://(.+\.replit\.dev|([a-z0-9-]+\.)?promptpics\.ai|([a-z0-9-]+\.)?app\.promptpics\.ai)",
    allow_credentials=False,  # frontend sends credentials: 'omit'
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app.on_event("startup")
def _ensure_tables() -> None:
    """Create tables on startup for local/dev usage."""
    try:
        # Clear CUDA cache on startup for clean memory state
        if torch.cuda.is_available():
            logger.info("Clearing CUDA cache on startup...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            logger.info(f"GPU Memory: {free_mem:.2f} GB free / {total_mem:.2f} GB total")

        Base.metadata.create_all(bind=engine)

        # Global lock to avoid concurrent GPU generations.
        generation_lock = threading.Lock()
        app.state.generation_lock = generation_lock
        set_generation_lock(generation_lock)

        # Optional prank-matcher warmup to avoid first-request latency.
        try:
            matcher = get_prank_matcher_llm()
            if matcher is not None:
                matcher.choose("warmup", ["warmup"])
        except Exception as exc:
            logger.warning("Prank matcher LLM warmup failed: %s", exc)
    except Exception as exc:
        logger.exception("Failed to create tables on startup")
        raise


# ---------------------------------------------------------------------------
# Security Middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def block_dotfiles(request: Request, call_next):
    """
    Deny access to dotfiles and VCS paths (e.g., /.git) defensively.
    """
    path = request.url.path
    if path.startswith("/.") or "/.git" in path:
        logger.warning("Blocked dotfile/VCS path: path=%s ip=%s", path, client_ip(request))
        return Response(status_code=404)
    # Light logging for obviously suspicious probes
    for marker in ("/wp-login", "/wp-admin", "/phpmyadmin", "/config.php", "/etc/passwd"):
        if marker in path:
            logger.warning("Suspicious probe blocked: path=%s ip=%s", path, client_ip(request))
            return Response(status_code=404)
    return await call_next(request)


# ---------------------------------------------------------------------------
# Include Routers
# ---------------------------------------------------------------------------

app.include_router(health.router)
app.include_router(admin.router)
app.include_router(generate.router)
app.include_router(pranks.router)
app.include_router(bench.router)
