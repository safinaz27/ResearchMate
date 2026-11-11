"""ORM models: documents and their chunks.

SQLite is the **source of truth**. Chroma (vectors) and BM25 (keywords) are
indexes that can be rebuilt from the ``chunks`` table, and citations
(document name + page/section) are built from these rows, never from LLM text.

Only portable column types are used (UUIDs as ``String(36)``, enums as
``VARCHAR`` + ``CHECK``, no arrays), so moving to PostgreSQL later only needs a
different ``DATABASE_URL``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_id() -> str:
    """A random UUID4 as a 36-character string (portable primary key)."""
    return str(uuid.uuid4())


def utc_now() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """Timezone-aware UTC datetimes on every database.

    SQLite has no timezone support and returns naive datetimes. This type
    rejects naive input, stores UTC, and re-attaches UTC when reading, so the
    application always sees aware datetimes (same as on PostgreSQL).
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("Naive datetime given; use timezone-aware UTC datetimes.")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class DocumentStatus(StrEnum):
    """Ingestion lifecycle: pending -> processing -> ready | failed."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class FileType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"


def _string_enum(enum_cls: type[StrEnum], name: str) -> Enum:
    """Store an enum as VARCHAR + CHECK constraint (portable; no native DB enum)."""
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=20,
        values_callable=lambda cls: [member.value for member in cls],  # store "ready", not "READY"
        validate_strings=True,
    )


class Base(DeclarativeBase):
    pass


class Document(Base):
    """An uploaded file and its ingestion status."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    filename: Mapped[str] = mapped_column(String(255))          # original name, shown in citations
    file_type: Mapped[FileType] = mapped_column(_string_enum(FileType, "ck_documents_file_type"))
    file_path: Mapped[str] = mapped_column(String(1024))        # where the upload is stored
    file_size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)  # duplicate detection
    status: Mapped[DocumentStatus] = mapped_column(
        _string_enum(DocumentStatus, "ck_documents_status"), default=DocumentStatus.PENDING
    )
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    page_count: Mapped[int | None] = mapped_column(Integer, default=None)  # None for DOCX
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, onupdate=utc_now)

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,  # let the DB's ON DELETE CASCADE remove chunks
        order_by="Chunk.chunk_index",
    )

    __table_args__ = (
        CheckConstraint("file_size_bytes >= 0", name="ck_documents_size_nonneg"),
        CheckConstraint("chunk_count >= 0", name="ck_documents_chunk_count_nonneg"),
    )

    def __repr__(self) -> str:
        return f"Document(id={self.id!r}, filename={self.filename!r}, status={self.status!r})"


class Chunk(Base):
    """A piece of a document's text, with the location used for citations.

    Exactly one of ``page_number`` (PDF, 1-based) or ``section_number`` (DOCX,
    1-based) is set. DOCX files have no real pages, so we never invent one;
    a CHECK constraint enforces this rule in the database itself.
    """

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)  # order within the document, from 0
    text: Mapped[str] = mapped_column(Text)
    page_number: Mapped[int | None] = mapped_column(Integer, default=None)
    section_number: Mapped[int | None] = mapped_column(Integer, default=None)
    token_count: Mapped[int] = mapped_column(Integer)

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
        CheckConstraint(
            "(page_number IS NULL) <> (section_number IS NULL)",
            name="ck_chunks_exactly_one_location",
        ),
        CheckConstraint("page_number IS NULL OR page_number >= 1", name="ck_chunks_page_pos"),
        CheckConstraint("section_number IS NULL OR section_number >= 1",
                        name="ck_chunks_section_pos"),
        CheckConstraint("chunk_index >= 0", name="ck_chunks_index_nonneg"),
        CheckConstraint("token_count >= 0", name="ck_chunks_tokens_nonneg"),
    )

    @property
    def location_label(self) -> str:
        """Human-readable location for citations: ``"p. 3"`` or ``"section 2"``."""
        if self.page_number is not None:
            return f"p. {self.page_number}"
        return f"section {self.section_number}"

    def __repr__(self) -> str:
        return (f"Chunk(id={self.id!r}, document_id={self.document_id!r}, "
                f"index={self.chunk_index}, {self.location_label})")
