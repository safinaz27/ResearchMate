"""Repositories: the only place that writes SQL queries for documents/chunks.

Routes and services call these methods instead of building queries, so
storage details stay in one module.

Repositories **flush but never commit**. The caller owns the transaction
(see ``session_scope``), so several repository calls can succeed or fail
together, e.g. "insert all chunks and mark the document ready" is atomic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from researchmate.db.models import Chunk, Document, DocumentStatus, FileType


@dataclass(frozen=True, slots=True)
class NewChunk:
    """Input for creating a chunk (produced by the chunker in a later step)."""

    chunk_index: int
    text: str
    token_count: int
    page_number: int | None = None
    section_number: int | None = None


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        *,
        filename: str,
        file_type: FileType,
        file_path: str,
        file_size_bytes: int,
        sha256: str,
    ) -> Document:
        """Insert a document with status ``pending``; ``id`` is available immediately."""
        document = Document(
            filename=filename,
            file_type=file_type,
            file_path=file_path,
            file_size_bytes=file_size_bytes,
            sha256=sha256,
            status=DocumentStatus.PENDING,
        )
        self.session.add(document)
        self.session.flush()  # sends the INSERT so defaults (id, timestamps) are set
        return document

    def get(self, document_id: str) -> Document | None:
        return self.session.get(Document, document_id)

    def get_by_sha256(self, sha256: str) -> Document | None:
        """Find an already-uploaded file with identical content."""
        stmt = select(Document).where(Document.sha256 == sha256).limit(1)
        return self.session.scalars(stmt).first()

    def list_all(self) -> list[Document]:
        """All documents, newest first."""
        stmt = select(Document).order_by(Document.created_at.desc(), Document.id)
        return list(self.session.scalars(stmt))

    def set_status(
        self,
        document: Document,
        status: DocumentStatus,
        *,
        error_message: str | None = None,
        page_count: int | None = None,
    ) -> None:
        """Move a document through its lifecycle.

        ``error_message`` is kept only for ``failed``; any other status clears
        it, so a successful retry does not keep showing an old error.
        """
        if status is DocumentStatus.FAILED and not error_message:
            raise ValueError("A failed document needs an error_message.")
        document.status = status
        document.error_message = error_message if status is DocumentStatus.FAILED else None
        if page_count is not None:
            document.page_count = page_count
        self.session.flush()

    def delete(self, document: Document) -> None:
        """Delete a document; its chunks are removed by ON DELETE CASCADE."""
        self.session.delete(document)
        self.session.flush()


class ChunkRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_chunks(self, document: Document, chunks: Sequence[NewChunk]) -> list[Chunk]:
        """Insert chunks for ``document`` and update its ``chunk_count``."""
        rows = [
            Chunk(
                document_id=document.id,
                chunk_index=c.chunk_index,
                text=c.text,
                token_count=c.token_count,
                page_number=c.page_number,
                section_number=c.section_number,
            )
            for c in chunks
        ]
        self.session.add_all(rows)
        document.chunk_count += len(rows)
        self.session.flush()
        return rows

    def list_for_document(self, document_id: str) -> list[Chunk]:
        stmt = (select(Chunk).where(Chunk.document_id == document_id)
                .order_by(Chunk.chunk_index))
        return list(self.session.scalars(stmt))

    def get_by_ids(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        """Load chunks (with their document) in the **same order** as ``chunk_ids``.

        Retrieval returns ranked IDs; keeping that order matters. Unknown IDs
        are skipped (e.g. a stale index entry for a deleted document).
        """
        if not chunk_ids:
            return []
        stmt = (select(Chunk).options(joinedload(Chunk.document))
                .where(Chunk.id.in_(chunk_ids)))
        by_id = {chunk.id: chunk for chunk in self.session.scalars(stmt)}
        return [by_id[i] for i in chunk_ids if i in by_id]

    def list_ready(self) -> list[Chunk]:
        """All chunks of ``ready`` documents (input for rebuilding BM25/Chroma)."""
        stmt = (
            select(Chunk)
            .join(Chunk.document)
            .options(joinedload(Chunk.document))
            .where(Document.status == DocumentStatus.READY)
            .order_by(Chunk.document_id, Chunk.chunk_index)
        )
        return list(self.session.scalars(stmt))
