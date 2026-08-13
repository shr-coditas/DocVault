import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.document import DocumentSearchStatus, DocumentVisibility
from app.models.document_grant import PrincipalType


class DocumentUpdate(BaseModel):
    """Rename and/or move. folder_id=null moves to the workspace root."""

    title: str | None = Field(default=None, min_length=1, max_length=255)
    folder_id: uuid.UUID | None = None


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    workspace_id: uuid.UUID
    folder_id: uuid.UUID | None
    owner_id: uuid.UUID
    title: str
    file_name: str
    mime_type: str
    size_bytes: int
    checksum_sha256: str
    visibility: DocumentVisibility
    search_status: DocumentSearchStatus
    created_at: datetime
    deleted_at: datetime | None


class VisibilityUpdate(BaseModel):
    """Set who can see the document."""

    visibility: DocumentVisibility


class GrantCreate(BaseModel):
    """Grant a user or team access to a private/team document."""

    principal_type: PrincipalType
    principal_id: uuid.UUID


class GrantOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    principal_type: PrincipalType
    principal_id: uuid.UUID
    granted_by: uuid.UUID
    granted_at: datetime
