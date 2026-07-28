import uuid
from datetime import datetime
from enum import StrEnum

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class PrincipalType(StrEnum):
    """What kind of thing a grant is given to."""

    USER = "user"
    TEAM = "team"


class DocumentGrant(Base):
    """An explicit access grant on a document, for a single user or a team.

    A grant is binary: it means "this principal can see the document". *What*
    they may do with it is still governed by their workspace role; this only
    controls visibility for ``private``/``team`` documents. The composite PK
    keeps grants unique per (document, principal).
    """

    __tablename__ = "document_grants"
    __table_args__ = (
        # bare suffix: the metadata naming convention adds the ck_<table>_ prefix
        sa.CheckConstraint("principal_type in ('user', 'team')", name="principal_type"),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    principal_type: Mapped[str] = mapped_column(sa.String(10), primary_key=True)
    principal_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    granted_by: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("users.id"))
    granted_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now()
    )
