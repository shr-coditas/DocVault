"""add document visibility and grants

Revision ID: a1c7f3e29b84
Revises: 9e5f2a8b4c6d
Create Date: 2026-07-23 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a1c7f3e29b84"
down_revision: str | None = "9e5f2a8b4c6d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # default 'workspace' keeps every existing document visible as it is today
    op.add_column(
        "documents",
        sa.Column(
            "visibility",
            sa.String(length=20),
            server_default="workspace",
            nullable=False,
        ),
    )
    # bare suffix: the metadata naming convention renders ck_documents_visibility
    op.create_check_constraint(
        "visibility",
        "documents",
        "visibility in ('private', 'team', 'workspace')",
    )

    op.create_table(
        "document_grants",
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("principal_type", sa.String(length=10), nullable=False),
        sa.Column("principal_id", sa.Uuid(), nullable=False),
        sa.Column("granted_by", sa.Uuid(), nullable=False),
        sa.Column(
            "granted_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "principal_type in ('user', 'team')",
            name=op.f("ck_document_grants_principal_type"),
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_document_grants_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["granted_by"],
            ["users.id"],
            name=op.f("fk_document_grants_granted_by_users"),
        ),
        sa.PrimaryKeyConstraint(
            "document_id",
            "principal_type",
            "principal_id",
            name=op.f("pk_document_grants"),
        ),
    )


def downgrade() -> None:
    op.drop_table("document_grants")
    op.drop_constraint("visibility", "documents", type_="check")
    op.drop_column("documents", "visibility")
