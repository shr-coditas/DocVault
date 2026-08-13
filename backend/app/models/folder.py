import uuid

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Folder(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Adjacency-list tree: each folder points at its parent (NULL = root).

    `postgresql_nulls_not_distinct` makes the unique constraint also apply to
    root folders - plain UNIQUE treats NULLs as distinct, which would allow
    duplicate root names.
    """

    __tablename__ = "folders"
    __table_args__ = (
        sa.UniqueConstraint(
            "workspace_id", "parent_id", "name", postgresql_nulls_not_distinct=True
        ),
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.ForeignKey("folders.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(sa.String(255))
    created_by: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("users.id"))
