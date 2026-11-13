"""Validating and storing uploaded files.

Everything about an upload is untrusted: the file name, the declared content
type and the bytes themselves. This module therefore:

* accepts only ``.pdf`` and ``.docx`` and checks that the **content** matches
  the extension (magic bytes), so a renamed ``.exe`` is rejected;
* never uses the client's file name as a path: files are stored as
  ``uploads/<document id>.<ext>``; the cleaned original name is only shown in
  citations;
* enforces the size limit while copying (it never trusts a declared size);
* hashes the content (SHA-256) while copying, to reject duplicate uploads;
* writes to a ``.part`` file first and renames it only after all checks pass,
  so a failed upload never leaves a half-written file behind.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from sqlalchemy.orm import Session

from researchmate.config import Settings
from researchmate.db.models import Document, FileType, new_id
from researchmate.db.repositories import DocumentRepository
from researchmate.errors import AppError

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS: dict[str, FileType] = {".pdf": FileType.PDF, ".docx": FileType.DOCX}
COPY_BLOCK_BYTES = 1024 * 1024
MAX_FILENAME_CHARS = 255
# The PDF spec allows a few junk bytes before the header; readers search the first 1 KiB.
PDF_HEADER_SEARCH_BYTES = 1024
# Parts every real .docx (Office Open XML word document) contains.
DOCX_REQUIRED_PARTS = frozenset({"[Content_Types].xml", "word/document.xml"})


class UnsupportedFileTypeError(AppError):
    status_code = 415
    code = "unsupported_file_type"


class FileTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"


class EmptyFileError(AppError):
    status_code = 400
    code = "empty_file"


class DuplicateDocumentError(AppError):
    status_code = 409
    code = "duplicate_document"


def sanitize_filename(raw: str | None) -> str:
    """Return a safe display name: last path component, no control characters.

    ``"../../etc/evil.pdf"`` and ``"C:\\x\\evil.pdf"`` both become ``"evil.pdf"``.
    The result is used for display and extension checks only, never as a path.
    """
    if not raw:
        return ""
    name = unicodedata.normalize("NFC", raw)
    name = PurePosixPath(name.replace("\\", "/")).name          # drop any directory part
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C")  # control chars
    name = re.sub(r"\s+", " ", name).strip().lstrip(".")     # no hidden-file names
    if len(name) > MAX_FILENAME_CHARS:                          # keep the extension
        stem, dot, ext = name.rpartition(".")
        keep = MAX_FILENAME_CHARS - len(ext) - len(dot)
        name = (stem[:keep] + dot + ext) if dot and keep > 0 else name[:MAX_FILENAME_CHARS]
    return name


def file_type_from_name(filename: str) -> FileType:
    """Map the extension (case-insensitive) to a FileType, or raise 415."""
    extension = Path(filename).suffix.lower()
    file_type = ALLOWED_EXTENSIONS.get(extension)
    if file_type is None:
        raise UnsupportedFileTypeError(
            f"Only PDF and DOCX files are supported (got {extension or 'no extension'!r}).",
            details={"allowed_extensions": sorted(ALLOWED_EXTENSIONS)},
        )
    return file_type


def verify_content(path: Path, file_type: FileType) -> None:
    """Check that the bytes really are the declared type (raise 415 if not).

    * PDF: ``%PDF-`` within the first 1 KiB.
    * DOCX: a valid ZIP archive containing the core Word parts. Only the ZIP
      directory is read here, nothing is decompressed.
    """
    if file_type is FileType.PDF:
        with path.open("rb") as f:
            head = f.read(PDF_HEADER_SEARCH_BYTES)
        if b"%PDF-" not in head:
            raise UnsupportedFileTypeError("The file has a .pdf extension but is not a PDF.")
        return

    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile:
        names = set()
    if not DOCX_REQUIRED_PARTS.issubset(names):
        raise UnsupportedFileTypeError(
            "The file has a .docx extension but is not a Word document."
        )


def _copy_with_limit(source: BinaryIO, target: Path, max_bytes: int) -> tuple[int, str]:
    """Copy ``source`` to ``target`` in blocks; return (size, sha256). Raise 413 over the limit."""
    digest = hashlib.sha256()
    size = 0
    with target.open("wb") as out:
        while block := source.read(COPY_BLOCK_BYTES):
            size += len(block)
            if size > max_bytes:
                raise FileTooLargeError(
                    f"File is larger than the {max_bytes // (1024 * 1024)} MB limit.",
                    details={"max_bytes": max_bytes},
                )
            digest.update(block)
            out.write(block)
    return size, digest.hexdigest()


def store_upload(
    source: BinaryIO,
    original_filename: str | None,
    *,
    settings: Settings,
    session: Session,
) -> Document:
    """Validate an uploaded file, store it, and create its ``pending`` document row.

    The caller commits the session (one transaction per request).

    Raises:
        UnsupportedFileTypeError: wrong extension, or content does not match it (415).
        FileTooLargeError: more than ``MAX_UPLOAD_MB`` (413).
        EmptyFileError: zero bytes (400).
        DuplicateDocumentError: identical content was already uploaded (409).
    """
    filename = sanitize_filename(original_filename)
    file_type = file_type_from_name(filename)  # cheapest check first, before any disk I/O

    document_id = new_id()
    relative_path = PurePosixPath("uploads") / f"{document_id}.{file_type.value}"
    final_path = settings.data_dir / relative_path
    part_path = final_path.with_name(f"{document_id}.part")
    repo = DocumentRepository(session)

    try:
        size, sha256 = _copy_with_limit(source, part_path, settings.max_upload_bytes)
        if size == 0:
            raise EmptyFileError("The uploaded file is empty.")
        verify_content(part_path, file_type)

        existing = repo.get_by_sha256(sha256)
        if existing is not None:
            raise DuplicateDocumentError(
                f"This file was already uploaded as {existing.filename!r}.",
                details={"existing_document_id": existing.id, "filename": existing.filename},
            )

        part_path.replace(final_path)  # atomic rename on the same drive
        document = repo.create(
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            file_path=relative_path.as_posix(),
            file_size_bytes=size,
            sha256=sha256,
        )
    except BaseException:
        # Never leave partial or orphaned files behind when anything fails.
        part_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        raise

    logger.info("Stored upload %s as %s (%s, %d bytes)", document.id, relative_path,
                file_type.value, size)
    return document
