"""add deleted_at to documents

Revision ID: 9e5f2a8b4c6d
Revises: 7c2d9e4f1a3b
Create Date: 2026-07-16 09:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "9e5f2a8b4c6d"
down_revision: str | None = "7c2d9e4f1a3b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "deleted_at")
