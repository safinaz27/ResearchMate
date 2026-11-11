"""Tests for the FastAPI skeleton: app factory, /health and JSON error handling."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from researchmate import __version__
from researchmate.api.dependencies import get_db_session
from researchmate.api.errors import AppError
from researchmate.api.main import create_app
from researchmate.db.models import FileType
from researchmate.db.repositories import DocumentRepository

FAKE_KEY = "gsk_test0000000000000000000000"


class _TeapotError(AppError):
    status_code = 418
    code = "teapot"


def _add_test_routes(app: FastAPI) -> None:
    """Routes that exist only in tests, to trigger each kind of error."""

    @app.get("/_test/app-error")
    def app_error() -> None:
        raise _TeapotError("I am a teapot", details={"hint": "use a kettle"})

    @app.get("/_test/crash")
    def crash() -> None:
        raise RuntimeError(f"secret internals {FAKE_KEY}")

    @app.get("/_test/items/{item_id}")
    def item(item_id: int) -> dict[str, int]:
        return {"item_id": item_id}

    @app.post("/_test/documents")
    def create_document(
        session: Annotated[Session, Depends(get_db_session)], fail: bool = False
    ) -> dict[str, str]:
        doc = DocumentRepository(session).create(
            filename="x.pdf", file_type=FileType.PDF, file_path="uploads/x.pdf",
            file_size_bytes=1, sha256="a" * 64,
        )
        if fail:
            raise RuntimeError("fail after insert")
        return {"id": doc.id}

    @app.get("/_test/documents/count")
    def count_documents(session: Annotated[Session, Depends(get_db_session)]) -> dict[str, int]:
        return {"count": len(DocumentRepository(session).list_all())}


@pytest.fixture
def client(make_settings, tmp_path: Path) -> Iterator[TestClient]:
    app = create_app(make_settings(data_dir=tmp_path / "data", environment="test"))
    _add_test_routes(app)
    # raise_server_exceptions=False: return the 500 response like a real server
    # instead of re-raising the exception into the test.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": "ResearchMate",
        "version": __version__,
        "environment": "test",
        "groq_configured": False,
    }


def test_health_reports_key_presence_but_never_the_key(make_settings, tmp_path: Path) -> None:
    app = create_app(make_settings(data_dir=tmp_path / "data", groq_api_key=FAKE_KEY))
    with TestClient(app) as test_client:
        response = test_client.get("/health")
    assert response.json()["groq_configured"] is True
    assert FAKE_KEY not in response.text


def test_startup_creates_data_directories(make_settings, tmp_path: Path) -> None:
    settings = make_settings(data_dir=tmp_path / "data")
    app = create_app(settings)
    assert not settings.data_dir.exists()  # creating the app has no side effects on disk
    with TestClient(app):
        pass
    for path in (settings.uploads_dir, settings.chroma_dir, settings.hf_cache_dir):
        assert path.is_dir()
    assert settings.sqlite_path.is_file()  # database + tables created at startup


def test_request_session_commits_on_success(client: TestClient) -> None:
    assert client.post("/_test/documents").status_code == 200
    assert client.get("/_test/documents/count").json() == {"count": 1}


def test_request_session_rolls_back_on_error(client: TestClient) -> None:
    assert client.post("/_test/documents", params={"fail": True}).status_code == 500
    assert client.get("/_test/documents/count").json() == {"count": 0}


def test_unknown_route_is_json_404(client: TestClient) -> None:
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found", "message": "Not Found"}}


def test_wrong_method_is_json_405_with_allow_header(client: TestClient) -> None:
    response = client.post("/health")
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"
    assert "GET" in response.headers["allow"]


def test_app_error_uses_its_status_code_and_details(client: TestClient) -> None:
    response = client.get("/_test/app-error")
    assert response.status_code == 418
    assert response.json() == {
        "error": {"code": "teapot", "message": "I am a teapot",
                  "details": {"hint": "use a kettle"}}
    }


def test_validation_error_is_json_422_without_echoed_input(client: TestClient) -> None:
    response = client.get("/_test/items/not-a-number")
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"][0]["loc"] == ["path", "item_id"]
    assert "input" not in error["details"][0]
    assert "not-a-number" not in response.text


def test_unexpected_error_is_generic_500_and_logged(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR, logger="researchmate.api.errors"):
        response = client.get("/_test/crash")
    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "internal_error", "message": "An unexpected error occurred."}
    }
    assert "secret internals" not in response.text  # no internals leak to the client
    assert any("Unhandled error on GET /_test/crash" in r.getMessage() for r in caplog.records)
