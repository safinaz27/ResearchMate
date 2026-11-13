"""FastAPI application factory.

Run the API (from the project root):

    .venv\\Scripts\\python.exe -m uvicorn researchmate.api.main:create_app --factory --port 8000

``--factory`` makes uvicorn call ``create_app()``. There is intentionally no
module-level ``app = create_app()``: importing this module must not read
``.env``, configure logging or create folders as a side effect.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from researchmate import __version__
from researchmate.api.errors import register_error_handlers
from researchmate.api.middleware import BodySizeLimitMiddleware
from researchmate.api.routes import documents, health
from researchmate.config import (
    Settings,
    configure_runtime_environment,
    ensure_directories,
    get_settings,
)
from researchmate.db.session import create_db_engine, create_session_factory, init_db
from researchmate.logging_config import configure_logging

logger = logging.getLogger(__name__)

# Room for multipart framing (boundaries, part headers, small form fields).
MULTIPART_OVERHEAD_BYTES = 64 * 1024


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
        # Starlette spools uploads > 1 MB to tempfile.gettempdir(), which is on C:
        # by default. Point Python's temp dir at data/tmp while the app runs.
        previous_tempdir = tempfile.tempdir
        tempfile.tempdir = str(settings.tmp_dir)
        engine = create_db_engine(settings)
        init_db(engine)
        app.state.session_factory = create_session_factory(engine)
        logger.info("%s %s started (environment=%s, data_dir=%s, database=%s)",
                    settings.app_name, __version__, settings.environment, settings.data_dir,
                    engine.url.render_as_string(hide_password=True))
        try:
            yield
        finally:
            # Close pooled connections; on Windows an open SQLite file can't be deleted.
            engine.dispose()
            tempfile.tempdir = previous_tempdir
            logger.info("%s stopped", settings.app_name)

    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.state.settings = settings
    register_error_handlers(app)
    app.add_middleware(BodySizeLimitMiddleware,
                       max_body_bytes=settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES)
    app.include_router(health.router)
    app.include_router(documents.router)
    return app
