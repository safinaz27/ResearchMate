"""Application error base class.

Lives outside the ``api`` package so business logic (ingestion, retrieval, ...)
can raise meaningful errors without importing FastAPI. The API layer maps every
``AppError`` to a JSON response using its ``status_code`` and ``code``.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for errors our own code raises on purpose.

    Subclasses set ``status_code`` (HTTP status) and ``code`` (stable,
    machine-readable string), e.g. an unsupported file type -> 415.
    """

    status_code: int = 400
    code: str = "bad_request"

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details
