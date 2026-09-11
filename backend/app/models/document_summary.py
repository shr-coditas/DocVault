import uuid

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class DocumentSummary(Base):
    """A short summary of the beginning of one immutable document."""

    __tablename__ = "document_summaries"
    __table_args__ = (
        sa.CheckConstraint(
            "length(btrim(summary)) > 0",
            name="summary_not_blank",
        ),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True,
    )
    summary: Mapped[str] = mapped_column(sa.Text)
