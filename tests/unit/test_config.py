"""Tests for researchmate.config."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from researchmate.config import (
    PROJECT_ROOT,
    configure_runtime_environment,
    ensure_directories,
)


def test_defaults_match_approved_plan(make_settings) -> None:
    s = make_settings()
    assert s.groq_base_url == "https://api.groq.com/openai/v1"
    assert s.groq_model == "openai/gpt-oss-120b"
    assert s.embedding_model == "BAAI/bge-small-en-v1.5"
    assert s.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert s.model_device == "cpu"
    assert s.rrf_k == 60
    assert s.groq_api_key is None
    assert s.groq_configured is False


def test_environment_variables_override_defaults(clean_env, make_settings) -> None:
    clean_env.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    clean_env.setenv("CHUNK_SIZE_TOKENS", "300")
    clean_env.setenv("LOG_JSON", "true")
    s = make_settings()
    assert s.groq_model == "openai/gpt-oss-20b"
    assert s.chunk_size_tokens == 300  # string from env converted to int
    assert s.log_json is True


def test_relative_data_dir_is_resolved_against_project_root(make_settings) -> None:
    s = make_settings(data_dir=Path("data"))
    assert s.data_dir == (PROJECT_ROOT / "data").resolve()
    assert s.sqlite_path.parent == s.data_dir
    assert s.resolved_database_url.startswith("sqlite:///")


def test_explicit_database_url_wins(make_settings) -> None:
    url = "postgresql+psycopg://user:pw@localhost/researchmate"
    assert make_settings(database_url=url).resolved_database_url == url


def test_empty_values_in_env_mean_not_configured(clean_env, make_settings) -> None:
    clean_env.setenv("GROQ_API_KEY", "   ")
    clean_env.setenv("DATABASE_URL", "")
    s = make_settings()
    assert s.groq_api_key is None
    assert s.database_url is None


def test_api_key_is_hidden_in_repr_and_dump(make_settings) -> None:
    secret = "gsk_" + "x" * 40
    s = make_settings(groq_api_key=secret)
    assert s.groq_configured is True
    assert secret not in repr(s)
    assert secret not in str(s.model_dump())
    assert s.groq_api_key.get_secret_value() == secret


def test_overlap_must_be_smaller_than_chunk_size(make_settings) -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP_TOKENS"):
        make_settings(chunk_size_tokens=100, chunk_overlap_tokens=100)


@pytest.mark.parametrize(
    "field, value",
    [("chunk_size_tokens", 10), ("rerank_top_n", 0), ("log_level", "LOUD"),
     ("model_device", "tpu"), ("max_upload_mb", 0)],
)
def test_invalid_values_are_rejected(make_settings, field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{field: value})


def test_ensure_directories_creates_layout(make_settings, tmp_path: Path) -> None:
    s = make_settings(data_dir=tmp_path / "data")
    ensure_directories(s)
    for path in (s.data_dir, s.uploads_dir, s.chroma_dir, s.hf_cache_dir):
        assert path.is_dir()


def test_runtime_environment_points_hf_cache_into_data_dir(
    make_settings, clean_env, tmp_path: Path
) -> None:
    clean_env.delenv("HF_HOME", raising=False)
    clean_env.delenv("ANONYMIZED_TELEMETRY", raising=False)
    s = make_settings(data_dir=tmp_path / "data")
    configure_runtime_environment(s)
    assert os.environ["HF_HOME"] == str(s.hf_cache_dir)
    assert os.environ["ANONYMIZED_TELEMETRY"] == "False"


def test_runtime_environment_respects_explicit_shell_value(
    make_settings, clean_env, tmp_path: Path
) -> None:
    clean_env.setenv("HF_HOME", str(tmp_path / "custom"))
    configure_runtime_environment(make_settings(data_dir=tmp_path / "data"))
    assert os.environ["HF_HOME"] == str(tmp_path / "custom")
