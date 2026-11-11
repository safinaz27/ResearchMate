"""Creating the Groq client and choosing a model.

Groq exposes an OpenAI-compatible REST API, so we use the official ``openai``
SDK pointed at Groq's base URL instead of adding a Groq-specific dependency.

This module is the **only** place where the API key's secret value is read
(``SecretStr.get_secret_value()``). Everything else passes ``Settings`` around,
whose ``repr`` masks the key.
"""

from __future__ import annotations

from collections.abc import Iterable

from openai import OpenAI

from researchmate.config import Settings


class GroqNotConfiguredError(RuntimeError):
    """Raised when an LLM call is attempted without ``GROQ_API_KEY``."""


def create_groq_client(settings: Settings, *, max_retries: int | None = None) -> OpenAI:
    """Return an OpenAI SDK client configured for Groq.

    Args:
        settings: Application settings (key, base URL, timeout, retries).
        max_retries: Override ``settings.llm_max_retries``, e.g. ``0`` for a
            diagnostic script that should fail fast instead of retrying.

    Raises:
        GroqNotConfiguredError: If ``GROQ_API_KEY`` is missing or empty.
    """
    if settings.groq_api_key is None:
        raise GroqNotConfiguredError(
            "GROQ_API_KEY is not set. Add it to the project's .env file."
        )
    return OpenAI(
        api_key=settings.groq_api_key.get_secret_value(),
        base_url=settings.groq_base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries if max_retries is None else max_retries,
    )


def select_model(available: Iterable[str], preferred: str, fallback: str) -> str | None:
    """Pick the model to use from the IDs reported by the live ``/models`` endpoint.

    Returns ``preferred`` if Groq lists it, otherwise ``fallback`` if listed,
    otherwise ``None`` (the caller decides how to report that).
    """
    ids = set(available)
    if preferred in ids:
        return preferred
    if fallback in ids:
        return fallback
    return None
