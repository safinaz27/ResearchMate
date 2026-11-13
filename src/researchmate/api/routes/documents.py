"""Document endpoints: upload, list, get."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.orm import Session

from researchmate.api.dependencies import get_app_settings, get_db_session
from researchmate.api.schemas import DocumentOut
from researchmate.config import Settings
from researchmate.db.repositories import DocumentRepository
from researchmate.errors import AppError
from researchmate.ingestion.upload import store_upload

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentNotFoundError(AppError):
    status_code = 404
    code = "document_not_found"


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: Annotated[UploadFile, File(description="A PDF or DOCX file")],
    settings: Annotated[Settings, Depends(get_app_settings)],
    session: Annotated[Session, Depends(get_db_session)],
) -> DocumentOut:
    """Upload a PDF or DOCX file. It is stored with status ``pending``.

    A sync ``def`` route: FastAPI runs it in a worker thread, so the blocking
    file copy and database calls don't stall the event loop.
    """
    document = store_upload(file.file, file.filename, settings=settings, session=session)
    return DocumentOut.model_validate(document)


@router.get("", response_model=list[DocumentOut])
def list_documents(
    session: Annotated[Session, Depends(get_db_session)],
) -> list[DocumentOut]:
    """All documents, newest first."""
    return [DocumentOut.model_validate(d) for d in DocumentRepository(session).list_all()]


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(
    document_id: str,
    session: Annotated[Session, Depends(get_db_session)],
) -> DocumentOut:
    document = DocumentRepository(session).get(document_id)
    if document is None:
        raise DocumentNotFoundError(f"No document with id {document_id!r}.")
    return DocumentOut.model_validate(document)
