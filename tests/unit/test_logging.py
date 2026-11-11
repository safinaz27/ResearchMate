"""Tests for researchmate.logging_config."""

from __future__ import annotations

import json
import logging

import pytest

from researchmate import logging_config
from researchmate.logging_config import REDACTED, configure_logging, redact

FAKE_GROQ_KEY = "gsk_" + "A1b2C3d4" * 6  # looks like a Groq key, is not one


@pytest.fixture(autouse=True)
def _reset_logging():
    """Restore global logging state after each test."""
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    saved_secrets = set(logging_config._known_secrets)
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
    logging_config._known_secrets.clear()
    logging_config._known_secrets.update(saved_secrets)


def _last_line(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().err.strip().splitlines()[-1]


def test_redact_masks_key_patterns() -> None:
    assert FAKE_GROQ_KEY not in redact(f"key={FAKE_GROQ_KEY}")
    assert redact("Authorization: Bearer abcdef123456") == f"Authorization: Bearer {REDACTED}"
    assert redact("nothing secret here") == "nothing secret here"


def test_json_output_is_valid_and_contains_extra_fields(make_settings, capsys) -> None:
    configure_logging(make_settings(log_json=True))
    logging.getLogger("researchmate.test").info("indexed", extra={"chunks": 42})
    record = json.loads(_last_line(capsys))
    assert record["level"] == "INFO"
    assert record["logger"] == "researchmate.test"
    assert record["message"] == "indexed"
    assert record["chunks"] == 42
    assert "ts" in record


def test_configured_key_is_redacted_everywhere(make_settings, capsys) -> None:
    # A key with an unusual format: only exact-value registration can catch it.
    odd_key = "custom-provider-secret-value-123"
    configure_logging(make_settings(log_json=True, groq_api_key=odd_key))
    log = logging.getLogger("researchmate.test")

    log.info("message %s", odd_key)                       # via format args
    log.info("extra field", extra={"header": odd_key})    # via extra
    try:
        raise RuntimeError(f"failed with {FAKE_GROQ_KEY}")
    except RuntimeError:
        log.exception("boom")                             # via traceback

    output = capsys.readouterr().err
    assert odd_key not in output
    assert FAKE_GROQ_KEY not in output
    assert output.count(REDACTED) >= 3


def test_text_format_is_human_readable(make_settings, capsys) -> None:
    configure_logging(make_settings(log_json=False))
    logging.getLogger("researchmate.test").warning("disk low", extra={"free_gb": 4.8})
    line = _last_line(capsys)
    assert "WARNING" in line and "disk low" in line and "free_gb=4.8" in line


def test_configure_logging_is_idempotent(make_settings) -> None:
    settings = make_settings()
    configure_logging(settings)
    configure_logging(settings)
    ours = [h for h in logging.getLogger().handlers if h.get_name() == "researchmate"]
    assert len(ours) == 1


def test_log_level_is_applied(make_settings) -> None:
    configure_logging(make_settings(log_level="WARNING"))
    assert logging.getLogger().level == logging.WARNING
