"""Tests for document upload: validation, storage, duplicates, size limits."""

from __future__ import annotations

import asyncio
import io
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

import docx
import pymupdf
import pytest
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException

from researchmate.api.main import create_app
from researchmate.api.middleware import BodySizeLimitMiddleware
from researchmate.config import Settings
from researchmate.ingestion.upload import sanitize_filename

MB = 1024 * 1024


def make_pdf(text: str = "Hello ResearchMate") -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def make_docx(text: str = "Hello ResearchMate") -> bytes:
    document = docx.Document()
    document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


@pytest.fixture
def settings(make_settings, tmp_path: Path) -> Settings:
    return make_settings(data_dir=tmp_path / "data", max_upload_mb=1)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def upload(client: TestClient, name: str, content: bytes,
           content_type: str = "application/octet-stream"):
    return client.post("/documents", files={"file": (name, content, content_type)})


def stored_files(settings: Settings) -> list[str]:
    return sorted(p.name for p in settings.uploads_dir.iterdir())


# --- happy path ------------------------------------------------------------------

def test_upload_pdf_stores_file_and_pending_row(client: TestClient, settings: Settings) -> None:
    content = make_pdf()
    response = upload(client, "Attention Is All You Need.pdf", content, "application/pdf")
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["filename"] == "Attention Is All You Need.pdf"
    assert body["file_type"] == "pdf"
    assert body["status"] == "pending"
    assert body["file_size_bytes"] == len(content)
    assert body["chunk_count"] == 0
    assert "file_path" not in body and "sha256" not in body  # internals not exposed
    # Stored under a server-chosen name, byte-for-byte identical, no .part left behind.
    assert stored_files(settings) == [f"{body['id']}.pdf"]
    assert (settings.uploads_dir / f"{body['id']}.pdf").read_bytes() == content


def test_upload_docx(client: TestClient, settings: Settings) -> None:
    response = upload(client, "notes.DOCX", make_docx())  # extension is case-insensitive
    assert response.status_code == 201, response.text
    assert response.json()["file_type"] == "docx"
    assert stored_files(settings) == [f"{response.json()['id']}.docx"]


def test_list_and_get_documents(client: TestClient) -> None:
    doc_id = upload(client, "a.pdf", make_pdf("a")).json()["id"]
    assert [d["id"] for d in client.get("/documents").json()] == [doc_id]
    assert client.get(f"/documents/{doc_id}").json()["filename"] == "a.pdf"


def test_get_unknown_document_is_404(client: TestClient) -> None:
    response = client.get("/documents/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "document_not_found"


# --- rejected uploads -----------------------------------------------------------------

def test_duplicate_content_is_rejected_with_existing_id(client: TestClient,
                                                        settings: Settings) -> None:
    content = make_pdf()
    first = upload(client, "paper.pdf", content).json()
    response = upload(client, "renamed copy.pdf", content)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "duplicate_document"
    assert error["details"] == {"existing_document_id": first["id"], "filename": "paper.pdf"}
    assert stored_files(settings) == [f"{first['id']}.pdf"]  # second copy cleaned up


@pytest.mark.parametrize("name", ["notes.txt", "script.exe", "paper.doc", "noextension"])
def test_disallowed_extension_is_415(client: TestClient, settings: Settings, name: str) -> None:
    response = upload(client, name, b"some bytes")
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_file_type"
    assert stored_files(settings) == []


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("fake.pdf", b"MZ\x90\x00 this is really an exe"),
        ("fake.pdf", make_docx()),                       # a DOCX renamed to .pdf
        ("fake.docx", make_pdf()),                       # a PDF renamed to .docx
        ("fake.docx", b"PK\x03\x04 not a real zip"),     # ZIP signature, broken archive
    ],
    ids=["exe-as-pdf", "docx-as-pdf", "pdf-as-docx", "broken-zip-as-docx"],
)
def test_content_must_match_extension(client: TestClient, settings: Settings, name: str,
                                      content: bytes) -> None:
    response = upload(client, name, content)
    assert response.status_code == 415
    assert stored_files(settings) == []


def test_zip_that_is_not_a_word_document_is_415(client: TestClient) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("hello.txt", "not a docx")
    assert upload(client, "archive.docx", buffer.getvalue()).status_code == 415


def test_empty_file_is_400(client: TestClient, settings: Settings) -> None:
    response = upload(client, "empty.pdf", b"")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "empty_file"
    assert stored_files(settings) == []


def test_missing_file_field_is_422(client: TestClient) -> None:
    response = client.post("/documents", data={"not_file": "x"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_file_just_over_limit_is_413_from_upload_check(client: TestClient,
                                                       settings: Settings) -> None:
    # Body is under limit + multipart overhead, so it reaches the route; the copy loop rejects it.
    content = make_pdf() + b"\0" * (MB + 1)
    response = upload(client, "big.pdf", content)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"
    assert stored_files(settings) == []


def test_huge_body_is_413_from_middleware_before_parsing(client: TestClient,
                                                         settings: Settings) -> None:
    response = upload(client, "huge.pdf", b"%PDF-" + b"\0" * (2 * MB))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    assert stored_files(settings) == []


# --- filename handling -----------------------------------------------------------------

def test_client_path_in_filename_is_never_used(client: TestClient, settings: Settings) -> None:
    response = upload(client, "../../outside.pdf", make_pdf())
    assert response.status_code == 201
    assert response.json()["filename"] == "outside.pdf"
    assert not (settings.data_dir.parent / "outside.pdf").exists()
    assert stored_files(settings) == [f"{response.json()['id']}.pdf"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("paper.pdf", "paper.pdf"),
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("C:\\Users\\x\\evil.pdf", "evil.pdf"),
        ("bad\x00name\n.pdf", "badname.pdf"),
        ("  spaced   out .pdf ", "spaced out .pdf"),
        (".hidden.pdf", "hidden.pdf"),
        ("", ""),
        (None, ""),
    ],
)
def test_sanitize_filename(raw: str | None, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_truncates_but_keeps_extension() -> None:
    name = sanitize_filename("a" * 300 + ".pdf")
    assert len(name) == 255
    assert name.endswith(".pdf")


# --- temp files stay on the project drive ---------------------------------------------

def test_temp_dir_is_redirected_while_app_runs(settings: Settings) -> None:
    before = tempfile.tempdir
    with TestClient(create_app(settings)):
        assert Path(tempfile.gettempdir()) == settings.tmp_dir
        assert settings.tmp_dir.is_dir()
    assert tempfile.tempdir == before  # restored on shutdown


# --- middleware without Content-Length (streamed / chunked body) ------------------------

def test_middleware_counts_streamed_bytes_without_content_length() -> None:
    seen: list[int] = []

    async def app(scope, receive, send) -> None:  # reads the whole body, like a form parser
        while True:
            message = await receive()
            seen.append(len(message["body"]))
            if not message.get("more_body"):
                return

    chunks = iter([b"x" * 60, b"x" * 60, b"x" * 60])

    async def receive():
        body = next(chunks)
        return {"type": "http.request", "body": body, "more_body": True}

    async def send(message) -> None:  # pragma: no cover - never reached
        raise AssertionError("nothing should be sent")

    middleware = BodySizeLimitMiddleware(app, max_body_bytes=100)
    scope = {"type": "http", "method": "POST", "path": "/documents", "headers": []}
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(middleware(scope, receive, send))
    assert excinfo.value.status_code == 413
    assert seen == [60]  # aborted on the second chunk, before the app saw it
