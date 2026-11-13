"""Mapping exceptions to consistent JSON error responses.

Every error the API returns has the same shape, so clients (the Streamlit UI,
tests, curl) can handle errors in one way:

    {"error": {"code": "not_found", "message": "...", "details": ...}}

``code`` is a stable, machine-readable string; ``message`` is for humans;
``details`` is optional (e.g. the list of validation problems).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from researchmate.errors import AppError

logger = logging.getLogger(__name__)

# Stable codes for the HTTP errors that FastAPI/Starlette raise themselves.
_HTTP_STATUS_CODES: dict[int, str] = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    415: "unsupported_media_type",
}


def error_response(status_code: int, code: str, message: str, details: Any = None) -> JSONResponse:
    """Build the standard error body."""
    body: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        body["details"] = jsonable_encoder(details)
    return JSONResponse(status_code=status_code, content={"error": body})


async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    logger.info("Request failed: %s %s -> %s (%s)", request.method, request.url.path,
                exc.status_code, exc.code)
    return error_response(exc.status_code, exc.code, exc.message, exc.details)


async def _handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = _HTTP_STATUS_CODES.get(exc.status_code, "http_error")
    response = error_response(exc.status_code, code, str(exc.detail))
    if exc.headers:  # e.g. "Allow" on 405
        response.headers.update(exc.headers)
    return response


async def _handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Keep where/what failed; drop the echoed input, which may be large or sensitive.
    details = [{"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")}
               for e in exc.errors()]
    return error_response(422, "validation_error", "Request validation failed.", details)


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    # Full traceback goes to the (redacted) log; the client gets no internals.
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return error_response(500, "internal_error", "An unexpected error occurred.")


def register_error_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to ``app``."""
    app.add_exception_handler(AppError, _handle_app_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_error)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(Exception, _handle_unexpected_error)
