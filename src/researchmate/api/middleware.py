"""ASGI middleware that caps the size of request bodies.

Why this is needed: for multipart uploads, Starlette receives and spools the
**whole** file to a temp file *before* our route code runs. A size check in
the route alone would therefore only fire after a huge upload was already
written to disk. This middleware stops it while it is being received:

1. If ``Content-Length`` is declared and too large, reply 413 immediately
   without reading the body.
2. Otherwise (e.g. chunked transfer with no length), count bytes as they
   arrive and abort with 413 as soon as the limit is crossed.

This is "pure" ASGI middleware (not ``BaseHTTPMiddleware``) because it must
wrap the ``receive`` channel to count bytes while the body streams in.
"""

from __future__ import annotations

from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from researchmate.api.errors import error_response


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _content_length(scope)
        if declared is not None and declared > self.max_body_bytes:
            response = error_response(
                413, "payload_too_large", "Request body is too large.",
                details={"max_bytes": self.max_body_bytes},
            )
            await response(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    # Raised inside the app while it reads the body; FastAPI
                    # re-raises HTTPException, and our handler returns JSON 413.
                    raise HTTPException(status_code=413, detail="Request body is too large.")
            return message

        await self.app(scope, limited_receive, send)


def _content_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None  # malformed header: fall back to counting bytes
    return None
