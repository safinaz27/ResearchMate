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
