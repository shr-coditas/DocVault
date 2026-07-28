import uuid
from datetime import datetime
from enum import StrEnum

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class DocumentVisibility(StrEnum):
    """Who can see a document, on top of the workspace-role capability check.

    - ``workspace``: any workspace member (the default).
    - ``restricted``: the owner, workspace owners, and whoever the document has
      been shared with — a grant counts the same whether it names a user or a
      team (see ``DocumentGrant``).

    This started as ``private``/``team``/``workspace``, where a team grant only
    applied at ``team`` visibility. That asymmetry was a bug: it silently made
    team grants inert on a ``private`` document. Once grants became symmetric,
    ``private`` and ``team`` meant the same thing, so they were merged into
    ``restricted`` rather than left as two names for one behaviour.
    """

    RESTRICTED = "restricted"
    WORKSPACE = "workspace"


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A stored file: metadata row in Postgres, bytes in object storage.

    `storage_key` is the object's address in the bucket (see
    `storage_service.document_key`). The row and the object are written in the
    same use case; the row is the source of truth for listing and access.
    """

    __tablename__ = "documents"
    __table_args__ = (
        sa.Index("ix_documents_workspace_folder", "workspace_id", "folder_id"),
        # the metadata naming convention prefixes this with ck_<table>_, so the
        # name given here is the bare suffix — passing "ck_documents_visibility"
        # would render as ck_documents_ck_documents_visibility
        sa.CheckConstraint("visibility in ('restricted', 'workspace')", name="visibility"),
    )

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
    # default 'workspace' preserves the pre-sharing behavior for every existing row
    visibility: Mapped[str] = mapped_column(
        sa.String(20), default=DocumentVisibility.WORKSPACE, server_default="workspace"
    )
    # null = active, set = in trash (soft delete; restorable until permanent delete)
    deleted_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
