"""Logging setup: readable text or JSON lines, with secret redaction.

Uses only the standard library ``logging`` module. Every log line passes
through :func:`redact` as the final step of formatting, so an API key cannot
leak through a log message, a formatting argument, an ``extra`` field, or an
exception traceback.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any

from researchmate.config import Settings

# Patterns that look like credentials, regardless of where they appear.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"gsk_[A-Za-z0-9]{10,}"),                    # Groq API keys
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),                  # OpenAI-style keys
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}"),      # Authorization headers
)
REDACTED = "[REDACTED]"

# Exact secret values registered at startup (e.g. the configured GROQ_API_KEY),
# so a key with an unexpected format is still masked.
_known_secrets: set[str] = set()

# Attributes every LogRecord has; anything else came from ``extra={...}``.
_STANDARD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime", "taskName"}

# Marker so configure_logging() can find and replace only its own handler.
_HANDLER_NAME = "researchmate"


def register_secret(value: str | None) -> None:
    """Add an exact value that must never appear in logs."""
    if value and len(value) >= 8:  # very short strings would mask normal words
        _known_secrets.add(value)


def redact(text: str) -> str:
    """Mask known secret values and anything that looks like an API key."""
    for secret in _known_secrets:
        text = text.replace(secret, REDACTED)
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:  # keep the "Bearer " prefix, mask only the token
            text = pattern.sub(lambda m: m.group(1) + REDACTED, text)
        else:
            text = pattern.sub(REDACTED, text)
    return text


class RedactingTextFormatter(logging.Formatter):
    """Human-readable single-line format for local development."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        extras = _extra_fields(record)
        if extras:
            line += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        return redact(line)


class RedactingJsonFormatter(logging.Formatter):
    """One JSON object per line: easy to search, filter and ship to log tools."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_extra_fields(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # default=str: non-JSON values (Path, UUID, ...) are stringified, not fatal.
        return redact(json.dumps(payload, default=str, ensure_ascii=False))


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS}


def configure_logging(settings: Settings) -> None:
    """Configure the root logger. Safe to call more than once (idempotent)."""
    if settings.groq_api_key is not None:
        register_secret(settings.groq_api_key.get_secret_value())

    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.get_name() == _HANDLER_NAME:
            root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(
        RedactingJsonFormatter() if settings.log_json else RedactingTextFormatter()
    )
    root.addHandler(handler)
    root.setLevel(settings.log_level)

    # Third-party libraries that are very chatty at INFO level.
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers",
                  "huggingface_hub", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
