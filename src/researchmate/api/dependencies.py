"""FastAPI dependencies shared by the routes."""

from __future__ import annotations

from fastapi import Request

from researchmate.config import Settings


def get_app_settings(request: Request) -> Settings:
    """Return the Settings the app was created with.

    Routes use this instead of the global ``get_settings()``, so a test app
    built with custom settings is fully isolated from the real ``.env``.
    """
    return request.app.state.settings
