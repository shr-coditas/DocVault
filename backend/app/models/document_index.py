"""Small audit record for each document indexing attempt."""

import uuid
from datetime import datetime
from enum import StrEnum

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UUIDPrimaryKeyMixin


class IndexRunStatus(StrEnum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentIndexRun(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "document_index_runs"
    __table_args__ = (
        sa.CheckConstraint(
            "status in ('processing', 'completed', 'failed')",
            name="status",
        ),
        sa.Index(
            "ix_document_index_runs_document_started",
            "document_id",
            "started_at",
        ),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("documents.id", ondelete="CASCADE")
    )
    status: Mapped[str] = mapped_column(sa.String(20))
    error: Mapped[str | None] = mapped_column(sa.Text)
    started_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
    )
    completed_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
