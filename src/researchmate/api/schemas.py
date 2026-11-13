"""Response models (the public JSON shape of API objects)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from researchmate.db.models import DocumentStatus, FileType


class DocumentOut(BaseModel):
    """A document as returned by the API.

    Internal fields (stored file path, content hash) are deliberately not
    exposed: clients don't need them, and the path reveals server layout.
    """

    model_config = ConfigDict(from_attributes=True)  # build directly from the ORM object

    id: str
    filename: str
    file_type: FileType
    file_size_bytes: int
    status: DocumentStatus
    error_message: str | None
    page_count: int | None
    chunk_count: int
    created_at: datetime
    updated_at: datetime
