"""FastAPI application factory.

Run the API (from the project root):

    .venv\\Scripts\\python.exe -m uvicorn researchmate.api.main:create_app --factory --port 8000

``--factory`` makes uvicorn call ``create_app()``. There is intentionally no
module-level ``app = create_app()``: importing this module must not read
``.env``, configure logging or create folders as a side effect.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from researchmate import __version__
from researchmate.api.errors import register_error_handlers
from researchmate.api.routes import health
from researchmate.config import (
    Settings,
    configure_runtime_environment,
    ensure_directories,
    get_settings,
)
from researchmate.logging_config import configure_logging

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app.

    Args:
        settings: Settings to use. Tests pass their own; ``None`` means the
            process-wide settings from ``.env`` / environment variables.
    """
    settings = settings or get_settings()
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Runs once when the server starts (and once per TestClient context).
        configure_runtime_environment(settings)
        ensure_directories(settings)
        logger.info("%s %s started (environment=%s, data_dir=%s)", settings.app_name,
                    __version__, settings.environment, settings.data_dir)
        yield
        logger.info("%s stopped", settings.app_name)

    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.state.settings = settings
    register_error_handlers(app)
    app.include_router(health.router)
    return app
