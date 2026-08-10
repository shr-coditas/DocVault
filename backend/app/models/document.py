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


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A stored file: metadata row in Postgres, bytes in object storage.

    `storage_key` is the object's address in the bucket (see
    `storage_service.document_key`). The row and the object are written in the
    same use case; the row is the source of truth for listing and access.
    """

    __tablename__ = "documents"
    __table_args__ = (
        # Supports composite foreign keys from every indexing table, enforcing
        # that a denormalised workspace_id always matches its document.
        sa.UniqueConstraint("id", "workspace_id"),
        sa.Index("ix_documents_workspace_folder", "workspace_id", "folder_id"),
        # the metadata naming convention prefixes this with ck_<table>_, so the
        # name given here is the bare suffix - passing "ck_documents_visibility"
        # would render as ck_documents_ck_documents_visibility
        sa.CheckConstraint("visibility in ('restricted', 'workspace')", name="visibility"),
        sa.CheckConstraint("index_attempts >= 0", name="index_attempts_non_negative"),
        # the indexing script's selector: partial, so it stays cheap once most
        # documents are indexed and the predicate matches almost nothing
        sa.Index(
            "ix_documents_unindexed",
            "created_at",
            postgresql_where=sa.text("indexed = false"),
        ),
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

    # -- indexing state ----------------------------------------------------
    # `indexed` is the only thing the indexing script selects on. The rest is
    # diagnostics, except `index_generation` - see below.
    indexed: Mapped[bool] = mapped_column(default=False, server_default=sa.false())
    indexed_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    # last failure, truncated. Never holds document content.
    index_error: Mapped[str | None] = mapped_column(sa.Text)
    # monotonic; lets the script stop retrying a document it can never parse
    index_attempts: Mapped[int] = mapped_column(default=0, server_default="0")
    # Bumped on every successful (re)index. Retrieval requires
    # document_chunks.index_generation == documents.index_generation, so chunks
    # written under a superseded generation are invisible rather than merely
    # "probably deleted". Today the indexing transaction alone would guarantee
    # that; this exists so a future step-wise workflow can write chunks
    # incrementally and flip visibility atomically with one UPDATE.
    index_generation: Mapped[int] = mapped_column(default=0, server_default="0")
    # A same-dimensional model swap is still incompatible. Retrieval composes
    # this profile equality with generation equality so profiles never mix.
    active_embedding_profile: Mapped[str | None] = mapped_column(sa.String(255))
