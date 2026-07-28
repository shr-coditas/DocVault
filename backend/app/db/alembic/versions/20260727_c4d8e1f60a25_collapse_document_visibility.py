"""collapse document visibility to restricted/workspace

Revision ID: c4d8e1f60a25
Revises: a1c7f3e29b84
Create Date: 2026-07-27 12:00:00.000000

``private`` and ``team`` became behaviourally identical once user grants and
team grants were made symmetric — both mean "restricted, opened by grants". Two
values for one behaviour is a trap, so they are merged into ``restricted``.

The CHECK constraint is dropped before the data is rewritten and recreated
afterwards, so neither the old nor the new constraint is ever violated mid-flight.

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4d8e1f60a25"
down_revision: str | None = "a1c7f3e29b84"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("visibility", "documents", type_="check")
    op.execute(
        sa.text("UPDATE documents SET visibility = 'restricted' WHERE visibility <> 'workspace'")
    )
    op.create_check_constraint(
        "visibility",
        "documents",
        "visibility in ('restricted', 'workspace')",
    )


def downgrade() -> None:
    # 'restricted' maps back to 'private'; the original 'team' rows are not
    # recoverable, and under the symmetric-grant rule they behaved identically
    # to 'private' anyway.
    op.drop_constraint("visibility", "documents", type_="check")
    op.execute(
        sa.text("UPDATE documents SET visibility = 'private' WHERE visibility <> 'workspace'")
    )
    op.create_check_constraint(
        "visibility",
        "documents",
        "visibility in ('private', 'team', 'workspace')",
    )
