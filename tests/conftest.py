"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from researchmate.config import Settings


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove every environment variable that maps to a Settings field.

    Guarantees tests see code defaults, not values from the developer's shell.
    """
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    return monkeypatch


@pytest.fixture
def make_settings(clean_env: pytest.MonkeyPatch):
    """Factory for Settings that ignores the real .env file."""

    def _make(**overrides: object) -> Settings:
        return Settings(_env_file=None, **overrides)

    return _make


def count_words(text: str) -> int:
    """Fake token counter for tests: one token per whitespace-separated word.

    Like the real WordPiece tokenizer it is additive across whitespace, which
    is the property the chunker relies on.
    """
    return len(text.split())


@pytest.fixture
def token_counter():
    return count_words


@pytest.fixture(autouse=True)
def _forbid_real_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a test would load (or download) the real Hugging Face tokenizer.

    The test suite must run offline; tests that chunk text inject ``count_words``.
    """

    def _refuse(self) -> None:
        raise RuntimeError("Tests must inject a fake token counter (see conftest.token_counter).")

    monkeypatch.setattr("researchmate.ingestion.tokens.HuggingFaceTokenCounter._load", _refuse)
