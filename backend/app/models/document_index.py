"""Durable document-index run state and canonical structure nodes."""

import uuid
from datetime import datetime
from enum import StrEnum

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UUIDPrimaryKeyMixin


class IndexRunStatus(StrEnum):
    CLAIMED = "claimed"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STAGED = "staged"
    ACTIVE = "active"
    FAILED = "failed"


_RUN_STATUSES = ", ".join(f"'{status.value}'" for status in IndexRunStatus)


class DocumentIndexRun(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "document_index_runs"
    __table_args__ = (
        sa.ForeignKeyConstraint(
            ["document_id", "workspace_id"],
            ["documents.id", "documents.workspace_id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("document_id"),
        sa.CheckConstraint(f"status in ({_RUN_STATUSES})", name="status"),
        sa.Index("ix_document_index_runs_status_lease", "status", "lease_expires_at"),
    )

    workspace_id: Mapped[uuid.UUID]
    document_id: Mapped[uuid.UUID]
    extractor_profile: Mapped[str] = mapped_column(sa.String(255))
    normalizer_profile: Mapped[str] = mapped_column(sa.String(255))
    chunker_profile: Mapped[str] = mapped_column(sa.String(255))
    embedding_profile: Mapped[str] = mapped_column(sa.String(255))
    status: Mapped[str] = mapped_column(sa.String(20))
    lease_token: Mapped[uuid.UUID] = mapped_column(unique=True)
    lease_expires_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True))
    artifact_key: Mapped[str | None] = mapped_column(sa.String(1024))
    quality_metrics: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict)
    error: Mapped[str | None] = mapped_column(sa.Text)
    started_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now()
    )


class DocumentStructureNode(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "document_nodes"
    __table_args__ = (
        sa.ForeignKeyConstraint(
            ["document_id", "workspace_id"],
            ["documents.id", "documents.workspace_id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("run_id", "logical_path"),
        sa.CheckConstraint("ordinal >= 0", name="ordinal_non_negative"),
        sa.CheckConstraint("confidence >= 0 and confidence <= 1", name="confidence_range"),
        sa.Index("ix_document_nodes_parent_ordinal", "parent_id", "ordinal"),
        sa.Index("ix_document_nodes_document", "workspace_id", "document_id"),
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("document_index_runs.id", ondelete="CASCADE")
    )
    workspace_id: Mapped[uuid.UUID]
    document_id: Mapped[uuid.UUID]
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.ForeignKey("document_nodes.id", ondelete="CASCADE")
    )
    logical_path: Mapped[str] = mapped_column(sa.String(1024))
    ordinal: Mapped[int]
    node_type: Mapped[str] = mapped_column(sa.String(40))
    heading_level: Mapped[int | None]
    text: Mapped[str | None] = mapped_column(sa.Text)
    source_spans: Mapped[list[dict[str, object]]] = mapped_column(JSONB, default=list)
    attributes: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict)
    confidence: Mapped[float] = mapped_column(sa.Float, default=1.0)
    content_hash: Mapped[str] = mapped_column(sa.String(64))
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now()
    )
