import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


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
    created_at: datetime
