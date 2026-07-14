import uuid
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UUIDPrimaryKeyMixin


class AuditLog(UUIDPrimaryKeyMixin, Base):
    """Append-only audit trail.

    Deliberately no foreign keys: audit rows must survive the deletion of the
    workspace/user/resource they describe.
    """

    __tablename__ = "audit_logs"
    __table_args__ = (sa.Index("ix_audit_logs_workspace_created", "workspace_id", "created_at"),)

    workspace_id: Mapped[uuid.UUID | None]
    actor_id: Mapped[uuid.UUID | None]
    action: Mapped[str] = mapped_column(sa.String(100))
    resource_type: Mapped[str] = mapped_column(sa.String(50))
    resource_id: Mapped[uuid.UUID | None]
    request_id: Mapped[str | None] = mapped_column(sa.String(64))
    extra: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=sa.text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now()
    )
