"""Liveness endpoint."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from researchmate import __version__
from researchmate.api.dependencies import get_app_settings
from researchmate.config import Settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"]
    app: str
    version: str
    environment: str
    groq_configured: bool


@router.get("/health", response_model=HealthResponse)
def health(settings: Annotated[Settings, Depends(get_app_settings)]) -> HealthResponse:
    """Report that the process is up.

    Deliberately cheap: it loads no models, opens no database and does not
    call Groq, so it stays fast and can be polled often. ``groq_configured``
    only says whether a key is set (never the key, never a live check).
    """
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=__version__,
        environment=settings.environment,
        groq_configured=settings.groq_configured,
    )
