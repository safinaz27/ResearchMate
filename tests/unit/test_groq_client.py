"""Offline tests for Groq client creation and model selection (no network, no key)."""

from __future__ import annotations

import pytest

from researchmate.llm.groq_client import (
    GroqNotConfiguredError,
    create_groq_client,
    select_model,
)

FAKE_KEY = "gsk_test0000000000000000000000"


def test_missing_key_raises_clear_error(make_settings) -> None:
    with pytest.raises(GroqNotConfiguredError, match="GROQ_API_KEY"):
        create_groq_client(make_settings())


def test_empty_key_is_treated_as_missing(make_settings) -> None:
    with pytest.raises(GroqNotConfiguredError):
        create_groq_client(make_settings(groq_api_key="   "))


def test_client_uses_settings(make_settings) -> None:
    settings = make_settings(groq_api_key=FAKE_KEY, llm_timeout_seconds=12.5, llm_max_retries=4)
    client = create_groq_client(settings)  # constructing the client makes no HTTP call
    assert str(client.base_url).rstrip("/") == "https://api.groq.com/openai/v1"
    assert client.api_key == FAKE_KEY
    assert client.timeout == 12.5
    assert client.max_retries == 4


def test_max_retries_override(make_settings) -> None:
    client = create_groq_client(make_settings(groq_api_key=FAKE_KEY), max_retries=0)
    assert client.max_retries == 0


def test_settings_repr_does_not_expose_key(make_settings) -> None:
    settings = make_settings(groq_api_key=FAKE_KEY)
    assert FAKE_KEY not in repr(settings)
    assert FAKE_KEY not in str(settings.model_dump())


def test_default_models(make_settings) -> None:
    settings = make_settings()
    assert settings.groq_model == "openai/gpt-oss-120b"
    assert settings.groq_fallback_model == "openai/gpt-oss-20b"


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        (["openai/gpt-oss-120b", "openai/gpt-oss-20b"], "openai/gpt-oss-120b"),
        (["openai/gpt-oss-20b", "other"], "openai/gpt-oss-20b"),
        (["other"], None),
        ([], None),
    ],
)
def test_select_model(available: list[str], expected: str | None) -> None:
    assert select_model(available, "openai/gpt-oss-120b", "openai/gpt-oss-20b") == expected
