import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A stored file: metadata row in Postgres, bytes in object storage.

    `storage_key` is the object's address in the bucket (see
    `storage_service.document_key`). The row and the object are written in the
    same use case; the row is the source of truth for listing and access.
    """

    __tablename__ = "documents"
    __table_args__ = (sa.Index("ix_documents_workspace_folder", "workspace_id", "folder_id"),)

    # the composite index below covers workspace-scoped lookups
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("workspaces.id", ondelete="CASCADE")
    )
    # nullable = lives in the workspace root; CASCADE = deleting a folder deletes
    # its documents (the UI warns before a folder delete)
    folder_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.ForeignKey("folders.id", ondelete="CASCADE")
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("users.id"))
    title: Mapped[str] = mapped_column(sa.String(255))
    file_name: Mapped[str] = mapped_column(sa.String(255))
    mime_type: Mapped[str] = mapped_column(sa.String(255))
    size_bytes: Mapped[int] = mapped_column(sa.BigInteger)
    checksum_sha256: Mapped[str] = mapped_column(sa.String(64))
    storage_key: Mapped[str] = mapped_column(sa.String(1024), unique=True)
    # null = active, set = in trash (soft delete; restorable until permanent delete)
    deleted_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
