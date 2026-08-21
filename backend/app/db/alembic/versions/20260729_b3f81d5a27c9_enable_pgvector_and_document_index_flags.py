"""enable pgvector and add document index flags

Revision ID: b3f81d5a27c9
Revises: c4d8e1f60a25
Create Date: 2026-07-29 17:40:00.000000

The extension gets its own revision, ahead of any column that needs it, so it
can be round-tripped on its own. It is created and dropped here, symmetrically;
the revision that adds the ``vector`` column only owns the table.

Note the compose database image must be ``pgvector/pgvector:pg17`` - the
extension ships as a shared library, so plain ``postgres:17`` cannot run this.

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b3f81d5a27c9"
down_revision: str | None = "c4d8e1f60a25"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # every existing document starts unindexed, which is exactly right: the
    # indexing script will pick them all up on its first run
    op.add_column(
        "documents",
        sa.Column("indexed", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )
    op.add_column("documents", sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("documents", sa.Column("index_error", sa.Text(), nullable=True))
    op.create_index(
        "ix_documents_unindexed",
        "documents",
        ["created_at"],
        postgresql_where=sa.text("indexed = false"),
    )


def downgrade() -> None:
    op.drop_index("ix_documents_unindexed", table_name="documents")
    op.drop_column("documents", "index_error")
    op.drop_column("documents", "indexed_at")
    op.drop_column("documents", "indexed")
    op.execute("DROP EXTENSION IF EXISTS vector")
