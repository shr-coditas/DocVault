"""DocumentSummary table creation

Revision ID: a70ae7ab34f7
Revises: d5a91c47e802
Create Date: 2026-08-26 15:30:28.491721

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a70ae7ab34f7"
down_revision: str | None = "d5a91c47e802"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "document_summaries",
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "length(btrim(summary)) > 0",
            name="summary_not_blank",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_document_summaries_document_id_documents",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "document_id",
            name="pk_document_summaries",
        ),
    )


def downgrade() -> None:
    op.drop_table("document_summaries")
