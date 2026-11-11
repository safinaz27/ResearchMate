"""Tests for the ORM models, constraints, sessions and repositories (SQLite file in tmp)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session, sessionmaker

from researchmate.db.models import Chunk, Document, DocumentStatus, FileType
from researchmate.db.repositories import ChunkRepository, DocumentRepository, NewChunk
from researchmate.db.session import (
    create_db_engine,
    create_session_factory,
    init_db,
    session_scope,
)


@pytest.fixture
def engine(make_settings, tmp_path: Path) -> Iterator[Engine]:
    settings = make_settings(data_dir=tmp_path / "data")
    settings.data_dir.mkdir(parents=True)
    engine = create_db_engine(settings)
    init_db(engine)
    yield engine
    engine.dispose()  # release the file so pytest can delete tmp_path on Windows


@pytest.fixture
def factory(engine: Engine) -> sessionmaker[Session]:
    return create_session_factory(engine)


@pytest.fixture
def session(factory: sessionmaker[Session]) -> Iterator[Session]:
    with factory() as s:
        yield s


def _make_document(session: Session, name: str = "paper.pdf", sha: str = "a" * 64) -> Document:
    return DocumentRepository(session).create(
        filename=name, file_type=FileType.PDF, file_path=f"uploads/{name}",
        file_size_bytes=1234, sha256=sha,
    )


def _pdf_chunks(n: int) -> list[NewChunk]:
    return [NewChunk(chunk_index=i, text=f"chunk {i}", token_count=10, page_number=i + 1)
            for i in range(n)]


# --- schema ------------------------------------------------------------------

def test_init_db_creates_tables_and_is_idempotent(engine: Engine) -> None:
    init_db(engine)  # second call must not fail or drop anything
    assert set(inspect(engine).get_table_names()) == {"documents", "chunks"}


def test_sqlite_file_is_in_data_dir(engine: Engine, tmp_path: Path) -> None:
    assert engine.url.database is not None
    assert Path(engine.url.database).resolve() == (tmp_path / "data" / "researchmate.db")


def test_foreign_keys_pragma_is_on(engine: Engine) -> None:
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1


# --- documents ---------------------------------------------------------------

def test_create_document_defaults(session: Session) -> None:
    doc = _make_document(session)
    assert len(doc.id) == 36
    assert doc.status is DocumentStatus.PENDING
    assert doc.chunk_count == 0
    assert doc.page_count is None
    assert doc.created_at.tzinfo is not None


def test_timestamps_are_utc_aware_after_reload(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as s:
        doc_id = _make_document(s).id
    with factory() as s:
        doc = s.get(Document, doc_id)
        assert doc is not None
        assert doc.created_at.tzinfo == UTC  # SQLite returned naive; our type re-attached UTC


def test_naive_datetime_is_rejected(session: Session) -> None:
    doc = _make_document(session)
    doc.created_at = datetime(2026, 1, 1)  # no tzinfo
    with pytest.raises(StatementError, match="Naive datetime"):
        session.flush()


def test_status_is_stored_as_lowercase_string(session: Session) -> None:
    doc = _make_document(session)
    raw = session.execute(text("SELECT status, file_type FROM documents WHERE id = :i"),
                          {"i": doc.id}).one()
    assert tuple(raw) == ("pending", "pdf")


def test_invalid_status_is_rejected_by_database(session: Session) -> None:
    doc = _make_document(session)
    with pytest.raises(IntegrityError):  # CHECK constraint, not just Python validation
        session.execute(text("UPDATE documents SET status = 'bogus' WHERE id = :i"),
                        {"i": doc.id})


def test_set_status_lifecycle(session: Session) -> None:
    repo = DocumentRepository(session)
    doc = _make_document(session)
    with pytest.raises(ValueError, match="error_message"):
        repo.set_status(doc, DocumentStatus.FAILED)
    repo.set_status(doc, DocumentStatus.FAILED, error_message="parse error")
    assert doc.error_message == "parse error"
    repo.set_status(doc, DocumentStatus.READY, page_count=12)
    assert doc.status is DocumentStatus.READY
    assert doc.error_message is None  # cleared on success
    assert doc.page_count == 12


def test_get_by_sha256_and_list_all(session: Session) -> None:
    repo = DocumentRepository(session)
    first = _make_document(session, "a.pdf", sha="1" * 64)
    second = _make_document(session, "b.pdf", sha="2" * 64)
    assert repo.get_by_sha256("2" * 64) is second
    assert repo.get_by_sha256("f" * 64) is None
    assert {d.id for d in repo.list_all()} == {first.id, second.id}


# --- chunks ------------------------------------------------------------------

def test_add_chunks_updates_count_and_keeps_order(session: Session) -> None:
    doc = _make_document(session)
    ChunkRepository(session).add_chunks(doc, list(reversed(_pdf_chunks(3))))
    assert doc.chunk_count == 3
    chunks = ChunkRepository(session).list_for_document(doc.id)
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert chunks[0].location_label == "p. 1"


def test_docx_chunk_uses_section_label(session: Session) -> None:
    doc = _make_document(session, "notes.docx")
    [chunk] = ChunkRepository(session).add_chunks(
        doc, [NewChunk(chunk_index=0, text="x", token_count=1, section_number=2)]
    )
    assert chunk.location_label == "section 2"


@pytest.mark.parametrize(
    ("page", "section"),
    [(1, 1), (None, None), (0, None), (None, 0)],
    ids=["both", "neither", "page-zero", "section-zero"],
)
def test_chunk_location_constraint(session: Session, page: int | None,
                                   section: int | None) -> None:
    doc = _make_document(session)
    with pytest.raises(IntegrityError):
        ChunkRepository(session).add_chunks(
            doc, [NewChunk(chunk_index=0, text="x", token_count=1,
                           page_number=page, section_number=section)]
        )


def test_duplicate_chunk_index_is_rejected(session: Session) -> None:
    doc = _make_document(session)
    with pytest.raises(IntegrityError):
        ChunkRepository(session).add_chunks(doc, [_pdf_chunks(1)[0], _pdf_chunks(1)[0]])


def test_chunk_requires_existing_document(session: Session) -> None:
    session.add(Chunk(document_id="0" * 36, chunk_index=0, text="x", token_count=1,
                      page_number=1))
    with pytest.raises(IntegrityError):  # foreign key enforced (PRAGMA foreign_keys=ON)
        session.flush()


def test_delete_document_cascades_to_chunks_in_database(session: Session) -> None:
    doc = _make_document(session)
    ChunkRepository(session).add_chunks(doc, _pdf_chunks(2))
    session.commit()
    DocumentRepository(session).delete(doc)
    session.commit()
    assert session.execute(text("SELECT COUNT(*) FROM chunks")).scalar() == 0


def test_get_by_ids_keeps_requested_order_and_skips_unknown(session: Session) -> None:
    doc = _make_document(session)
    c0, c1, c2 = ChunkRepository(session).add_chunks(doc, _pdf_chunks(3))
    found = ChunkRepository(session).get_by_ids([c2.id, "missing", c0.id])
    assert [c.id for c in found] == [c2.id, c0.id]
    assert found[0].document.filename == "paper.pdf"  # document loaded for citations
    assert ChunkRepository(session).get_by_ids([]) == []


def test_list_ready_only_returns_chunks_of_ready_documents(session: Session) -> None:
    repo = DocumentRepository(session)
    ready = _make_document(session, "ready.pdf", sha="1" * 64)
    pending = _make_document(session, "pending.pdf", sha="2" * 64)
    ChunkRepository(session).add_chunks(ready, _pdf_chunks(2))
    ChunkRepository(session).add_chunks(pending, _pdf_chunks(1))
    repo.set_status(ready, DocumentStatus.READY)
    chunks = ChunkRepository(session).list_ready()
    assert [(c.document.filename, c.chunk_index) for c in chunks] == [
        ("ready.pdf", 0), ("ready.pdf", 1)
    ]


# --- transactions --------------------------------------------------------------

def test_session_scope_commits_on_success(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as s:
        doc_id = _make_document(s).id
    with factory() as s:
        assert s.get(Document, doc_id) is not None


def test_session_scope_rolls_back_on_error(factory: sessionmaker[Session]) -> None:
    with pytest.raises(RuntimeError), session_scope(factory) as s:
        doc = _make_document(s)
        ChunkRepository(s).add_chunks(doc, _pdf_chunks(1))  # flushed, not yet committed
        raise RuntimeError("boom")
    with factory() as s:
        assert s.execute(text("SELECT COUNT(*) FROM documents")).scalar() == 0
        assert s.execute(text("SELECT COUNT(*) FROM chunks")).scalar() == 0
