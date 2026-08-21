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
      been shared with - a grant counts the same whether it names a user or a
      team (see ``DocumentGrant``).

    This started as ``private``/``team``/``workspace``, where a team grant only
    applied at ``team`` visibility. That asymmetry was a bug: it silently made
    team grants inert on a ``private`` document. Once grants became symmetric,
    ``private`` and ``team`` meant the same thing, so they were merged into
    ``restricted`` rather than left as two names for one behaviour.
    """

    RESTRICTED = "restricted"
    WORKSPACE = "workspace"


class DocumentSearchStatus(StrEnum):
    """User-safe readiness state for document search and question answering."""

    READY = "ready"
    WAITING_FOR_INDEX = "waiting_for_index"
    INDEXING_FAILED = "indexing_failed"


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A stored file: metadata row in Postgres, bytes in object storage.

    `storage_key` is the object's address in the bucket (see
    `storage_service.document_key`). The row and the object are written in the
    same use case; the row is the source of truth for listing and access.
    """

    __tablename__ = "documents"
    __table_args__ = (
        sa.UniqueConstraint("id", "workspace_id"),
        sa.Index("ix_documents_workspace_folder", "workspace_id", "folder_id"),
        sa.CheckConstraint("visibility in ('restricted', 'workspace')", name="visibility"),
        sa.Index(
            "ix_documents_unindexed",
            "created_at",
            postgresql_where=sa.text("indexed = false"),
        ),
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("workspaces.id", ondelete="CASCADE")
    )

    folder_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.ForeignKey("folders.id", ondelete="CASCADE")
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("users.id"))
    title: Mapped[str] = mapped_column(sa.String(255))
    file_name: Mapped[str] = mapped_column(sa.String(255))
    mime_type: Mapped[str] = mapped_column(sa.String(255))
    size_bytes: Mapped[int] = mapped_column(sa.BigInteger)
    storage_key: Mapped[str] = mapped_column(sa.String(1024), unique=True)
    visibility: Mapped[str] = mapped_column(
        sa.String(20), default=DocumentVisibility.WORKSPACE, server_default="workspace"
    )
    deleted_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    indexed: Mapped[bool] = mapped_column(default=False, server_default=sa.false())
    indexed_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    index_error: Mapped[str | None] = mapped_column(sa.Text)

    @property
    def search_status(self) -> DocumentSearchStatus:
        """Expose readiness without leaking internal indexing diagnostics.

        A dirty or failed reindex may leave ``indexed`` false while the previous
        active generation remains intentionally searchable. The active profile
        and generation are therefore the stronger readiness signal.
        """
        if self.indexed:
            return DocumentSearchStatus.READY
        if self.index_error:
            return DocumentSearchStatus.INDEXING_FAILED
        return DocumentSearchStatus.WAITING_FOR_INDEX
