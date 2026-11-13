"""Tests for page-aware PDF/DOCX parsing and the ingestion pipeline status lifecycle."""

from __future__ import annotations

import io
import logging
import zipfile
from collections.abc import Iterator
from pathlib import Path

import docx
import pymupdf
import pytest
from fastapi.testclient import TestClient

from researchmate.api.main import create_app
from researchmate.config import Settings
from researchmate.db.models import FileType
from researchmate.ingestion import pipeline
from researchmate.ingestion.parsing import (
    ParseError,
    TextUnit,
    parse_document,
    parse_docx,
    parse_pdf,
)

BIG = 100 * 1024 * 1024


# --- builders for real sample files ---------------------------------------------------

def write_pdf(path: Path, pages: list[str | None]) -> Path:
    """One page per entry; ``None`` = blank page."""
    pdf = pymupdf.open()
    for text in pages:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()
    return path


def write_docx(path: Path, build) -> Path:
    document = docx.Document()
    build(document)
    document.save(path)
    return path


# --- PDF -------------------------------------------------------------------------------

def test_pdf_one_unit_per_page_with_1_based_numbers(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "p.pdf", ["first page", None, "third page"])
    parsed = parse_pdf(path)
    assert parsed.page_count == 3                          # blank page still counted
    assert [u.page_number for u in parsed.units] == [1, 3]  # ...but produces no unit
    assert "first page" in parsed.units[0].text
    assert "third page" in parsed.units[1].text
    assert all(u.section_number is None for u in parsed.units)


def test_pdf_keeps_content_order_for_two_columns(tmp_path: Path) -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    for y, left in [(100, "L1"), (120, "L2")]:
        page.insert_text((72, y), left)
    for y, right in [(100, "R1"), (120, "R2")]:
        page.insert_text((320, y), right)
    path = tmp_path / "cols.pdf"
    pdf.save(path)
    pdf.close()
    text = parse_pdf(path).units[0].text
    # Left column fully before the right one, not interleaved line by line.
    assert text.index("L1") < text.index("L2") < text.index("R1") < text.index("R2")


def test_pdf_password_protected(tmp_path: Path) -> None:
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "secret")
    path = tmp_path / "locked.pdf"
    pdf.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    pdf.close()
    with pytest.raises(ParseError, match="password-protected"):
        parse_pdf(path)


def test_pdf_damaged(tmp_path: Path) -> None:
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"%PDF-1.4 this is not really a pdf")
    with pytest.raises(ParseError, match="damaged"):
        parse_pdf(path)


def test_pdf_without_text_suggests_scanned(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "blank.pdf", [None, None])
    with pytest.raises(ParseError, match="scanned PDF"):
        parse_document(path, FileType.PDF, max_docx_uncompressed_bytes=BIG)


# --- DOCX --------------------------------------------------------------------------------

def _research_doc(d) -> None:
    d.add_paragraph("Intro before any heading.")
    d.add_heading("Methods", level=1)
    d.add_paragraph("We trained a model.")
    table = d.add_table(rows=2, cols=3)
    table.cell(0, 0).text = "Model"
    table.cell(0, 1).text = "F1"
    table.cell(1, 0).text = "Ours"
    table.cell(1, 1).text = "0.91"
    table.cell(0, 1).merge(table.cell(0, 2))  # merged header cell
    d.add_heading("Results", level=2)
    d.add_paragraph("")                        # empty paragraphs are skipped
    d.add_paragraph("It worked.")


def test_docx_sections_split_at_headings(tmp_path: Path) -> None:
    parsed = parse_docx(write_docx(tmp_path / "r.docx", _research_doc),
                        max_uncompressed_bytes=BIG)
    assert parsed.page_count is None
    assert [(u.section_number, u.title) for u in parsed.units] == [
        (1, None), (2, "Methods"), (3, "Results")
    ]
    assert all(u.page_number is None for u in parsed.units)  # never fake pages
    assert parsed.units[0].text == "Intro before any heading."
    assert parsed.units[1].text == "Methods\nWe trained a model.\nModel | F1\nOurs | 0.91 |"
    assert parsed.units[2].text == "Results\nIt worked."


def test_docx_starting_with_heading_has_it_as_section_1(tmp_path: Path) -> None:
    def build(d) -> None:
        d.add_heading("Title Here", level=0)  # "Title" style
        d.add_paragraph("Body.")

    parsed = parse_docx(write_docx(tmp_path / "t.docx", build), max_uncompressed_bytes=BIG)
    assert [(u.section_number, u.title, u.text) for u in parsed.units] == [
        (1, "Title Here", "Title Here\nBody.")
    ]


def test_docx_without_headings_is_one_section(tmp_path: Path) -> None:
    def build(d) -> None:
        for i in range(3):
            d.add_paragraph(f"Paragraph {i}.")

    parsed = parse_docx(write_docx(tmp_path / "n.docx", build), max_uncompressed_bytes=BIG)
    assert len(parsed.units) == 1
    assert parsed.units[0].section_number == 1


def test_empty_docx_has_no_text(tmp_path: Path) -> None:
    path = write_docx(tmp_path / "e.docx", lambda d: None)
    with pytest.raises(ParseError, match="No extractable text"):
        parse_document(path, FileType.DOCX, max_docx_uncompressed_bytes=BIG)


def test_docx_zip_bomb_guard(tmp_path: Path) -> None:
    path = write_docx(tmp_path / "b.docx", lambda d: d.add_paragraph("x"))
    with pytest.raises(ParseError, match="over the 0 MB limit"):
        parse_docx(path, max_uncompressed_bytes=1000)


def test_docx_damaged(tmp_path: Path) -> None:
    path = tmp_path / "broken.docx"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:  # passes the upload check, but no real Word XML
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", "not xml at all")
    path.write_bytes(buffer.getvalue())
    with pytest.raises(ParseError, match="damaged"):
        parse_docx(path, max_uncompressed_bytes=BIG)


def test_text_unit_needs_exactly_one_location() -> None:
    with pytest.raises(ValueError):
        TextUnit(text="x")
    with pytest.raises(ValueError):
        TextUnit(text="x", page_number=1, section_number=1)


# --- pipeline through the API ------------------------------------------------------------

@pytest.fixture
def settings(make_settings, tmp_path: Path) -> Settings:
    return make_settings(data_dir=tmp_path / "data")


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings), raise_server_exceptions=False) as test_client:
        yield test_client


def _upload(client: TestClient, path: Path):
    return client.post("/documents", files={"file": (path.name, path.read_bytes())})


def test_valid_pdf_becomes_ready_with_page_count(client: TestClient, tmp_path: Path) -> None:
    response = _upload(client, write_pdf(tmp_path / "ok.pdf", ["a", "b"]))
    assert response.status_code == 201
    body = response.json()
    assert (body["status"], body["page_count"], body["error_message"]) == ("ready", 2, None)
    assert client.get(f"/documents/{body['id']}").json()["status"] == "ready"  # persisted


def test_valid_docx_becomes_ready_without_page_count(client: TestClient,
                                                     tmp_path: Path) -> None:
    body = _upload(client, write_docx(tmp_path / "ok.docx", _research_doc)).json()
    assert (body["status"], body["page_count"]) == ("ready", None)


def test_unparseable_file_is_stored_as_failed(client: TestClient, tmp_path: Path) -> None:
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "secret")
    path = tmp_path / "locked.pdf"
    pdf.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    pdf.close()
    response = _upload(client, path)
    assert response.status_code == 201  # the upload itself succeeded
    body = response.json()
    assert body["status"] == "failed"
    assert body["error_message"] == "The PDF is password-protected."
    assert client.get(f"/documents/{body['id']}").json()["status"] == "failed"


def test_unexpected_parser_error_is_recorded_and_logged(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("internal detail")

    monkeypatch.setattr(pipeline, "parse_document", boom)
    with caplog.at_level(logging.ERROR, logger="researchmate.ingestion.pipeline"):
        body = _upload(client, write_pdf(tmp_path / "x.pdf", ["a"])).json()
    assert body["status"] == "failed"
    assert body["error_message"] == "Unexpected error while processing (RuntimeError)."
    assert "internal detail" not in body["error_message"]  # details only in the log
    assert any(r.exc_info for r in caplog.records)


def test_upload_is_committed_even_if_processing_crashes(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def crash(*args, **kwargs):
        raise RuntimeError("process died")

    monkeypatch.setattr("researchmate.api.routes.documents.process_document", crash)
    assert _upload(client, write_pdf(tmp_path / "y.pdf", ["a"])).status_code == 500
    [document] = client.get("/documents").json()  # the upload row survived the crash
    assert document["status"] == "pending"


def test_missing_stored_file_fails_cleanly(client: TestClient, settings: Settings,
                                           tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    original = pipeline._parse

    def delete_then_parse(document, s):
        (s.data_dir / document.file_path).unlink()
        return original(document, s)

    monkeypatch.setattr(pipeline, "_parse", delete_then_parse)
    body = _upload(client, write_pdf(tmp_path / "z.pdf", ["a"])).json()
    assert (body["status"], body["error_message"]) == ("failed", "The stored file is missing.")
