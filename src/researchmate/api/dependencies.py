"""FastAPI dependencies shared by the routes."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from researchmate.config import Settings
from researchmate.db.session import session_scope


def get_app_settings(request: Request) -> Settings:
    """Return the Settings the app was created with.

    Routes use this instead of the global ``get_settings()``, so a test app
    built with custom settings is fully isolated from the real ``.env``.
    """
    return request.app.state.settings


def get_db_session(request: Request) -> Iterator[Session]:
    """One database session per request: commit on success, rollback on error.

    The session factory is created at startup (see ``create_app``'s lifespan).
    """
    with session_scope(request.app.state.session_factory) as session:
        yield session
