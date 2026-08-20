"""add unsupported_evidence and answer_rejected message kinds

Revision ID: d5a91c47e802
Revises: 4f8c2d7a91e3
Create Date: 2026-08-18 21:40:00.000000

Purely additive: the ``kind`` check constraint gains two permitted values and
loses none, so every existing row still satisfies it and no data is rewritten.

The two kinds exist because a turn that retrieved successfully and still shows
no answer had, until now, to borrow a kind that misreported the reason.
``no_sources`` claims nothing was found when passages were found and graded as
not supporting an answer; ``generation_unavailable`` blames a provider that
either answered fine and had its draft rejected, or was never called at all.

Only the agentic query path writes either value, and that path is behind a
feature flag that is off by default - so this migration changes what the column
is *allowed* to hold, not what any current deployment puts in it.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "d5a91c47e802"
down_revision: str | None = "4f8c2d7a91e3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Frozen copies, as every migration keeps: this file imports nothing from `app`,
# so a later edit to `MessageKind` cannot silently rewrite what this revision did.
_KINDS_BEFORE = (
    "answer",
    "no_sources",
    "generation_unavailable",
    "refusal",
    "decline",
    "chitchat",
    "clarification",
    "scope_unavailable",
    "error",
)
_KINDS_AFTER = (*_KINDS_BEFORE, "unsupported_evidence", "answer_rejected")


def _kind_constraint(kinds: Sequence[str]) -> str:
    values = ", ".join(f"'{kind}'" for kind in kinds)
    return f"kind is null or kind in ({values})"


def upgrade() -> None:
    # Bare suffix, no op.f(): the metadata naming convention renders this as
    # ck_messages_kind, and wrapping it would opt out of the convention the
    # model still follows - leaving two names for one constraint.
    op.drop_constraint("kind", "messages", type_="check")
    op.create_check_constraint("kind", "messages", _kind_constraint(_KINDS_AFTER))


def downgrade() -> None:
    # Rows written by the agentic path would violate the narrower constraint, so
    # they are relabelled to the kind each previously borrowed rather than
    # deleted: the turn genuinely happened and its provenance ledger is intact.
    op.execute("UPDATE messages SET kind = 'no_sources' WHERE kind = 'unsupported_evidence'")
    op.execute("UPDATE messages SET kind = 'refusal' WHERE kind = 'answer_rejected'")
    op.drop_constraint("kind", "messages", type_="check")
    op.create_check_constraint("kind", "messages", _kind_constraint(_KINDS_BEFORE))
