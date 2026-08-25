"""add persistent conversation history

Revision ID: 4f8c2d7a91e3
Revises: e7a24c9f10b6
Create Date: 2026-08-13 16:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "4f8c2d7a91e3"
down_revision: str | None = "e7a24c9f10b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("created_by", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=160), nullable=False),
        sa.Column("scope_mode", sa.String(length=20), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "scope_mode in ('workspace', 'selected')",
            name="scope_mode",
        ),
        sa.ForeignKeyConstraint(
            ["created_by"],
            ["users.id"],
            name="fk_conversations_created_by_users",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name="fk_conversations_workspace_id_workspaces",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_conversations"),
    )
    op.create_index(
        "ix_conversations_creator_workspace_updated",
        "conversations",
        ["created_by", "workspace_id", "updated_at", "id"],
    )

    op.create_table(
        "conversation_documents",
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("title_snapshot", sa.String(length=255), nullable=False),
        sa.Column("file_name_snapshot", sa.String(length=255), nullable=False),
        sa.CheckConstraint("position >= 0 and position < 10", name="position_range"),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_conversation_documents_conversation_id_conversations",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "conversation_id",
            "document_id",
            name="pk_conversation_documents",
        ),
        sa.UniqueConstraint(
            "conversation_id",
            "position",
            name="uq_conversation_documents_conversation_position",
        ),
    )

    op.create_table(
        "messages",
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("kind", sa.String(length=40), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("sequence > 0", name="sequence_positive"),
        sa.CheckConstraint("role in ('user', 'assistant')", name="role"),
        sa.CheckConstraint(
            "status in ('pending', 'complete', 'failed')",
            name="status",
        ),
        sa.CheckConstraint(
            "kind is null or kind in "
            "('answer', 'no_sources', 'generation_unavailable', 'refusal', "
            "'decline', 'chitchat', 'clarification', 'scope_unavailable', 'error')",
            name="kind",
        ),
        sa.CheckConstraint(
            "input_tokens is null or input_tokens >= 0",
            name="input_tokens_non_negative",
        ),
        sa.CheckConstraint(
            "output_tokens is null or output_tokens >= 0",
            name="output_tokens_non_negative",
        ),
        sa.CheckConstraint(
            "(role = 'user' and status = 'complete' and kind is null "
            "and content is not null) or "
            "(role = 'assistant' and status in ('complete', 'failed') "
            "and kind is not null and content is not null)",
            name="lifecycle",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_messages_conversation_id_conversations",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_messages"),
        sa.UniqueConstraint(
            "conversation_id",
            "sequence",
            name="uq_messages_conversation_sequence",
        ),
    )

    op.create_table(
        "message_sources",
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        sa.Column("document_title_snapshot", sa.String(length=255), nullable=False),
        sa.Column("section_path", sa.Text(), nullable=True),
        sa.Column("chunk_type", sa.String(length=20), nullable=False),
        sa.Column("retrieval_rank", sa.Integer(), nullable=False),
        sa.Column(
            "supplied_to_model",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        sa.Column("citation_marker", sa.Integer(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.CheckConstraint("retrieval_rank > 0", name="retrieval_rank_positive"),
        sa.CheckConstraint(
            "citation_marker is null or citation_marker > 0",
            name="citation_marker_positive",
        ),
        sa.CheckConstraint(
            "citation_marker is null or supplied_to_model",
            name="citation_requires_supplied",
        ),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["messages.id"],
            name="fk_message_sources_message_id_messages",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_message_sources"),
        sa.UniqueConstraint(
            "message_id",
            "retrieval_rank",
            name="uq_message_sources_message_rank",
        ),
        sa.UniqueConstraint(
            "message_id",
            "chunk_id",
            name="uq_message_sources_message_chunk",
        ),
        sa.UniqueConstraint(
            "message_id",
            "citation_marker",
            name="uq_message_sources_message_citation",
        ),
    )
    op.create_index(
        "ix_message_sources_document",
        "message_sources",
        ["document_id"],
    )


def downgrade() -> None:
    op.drop_table("message_sources")
    op.drop_table("messages")
    op.drop_table("conversation_documents")
    op.drop_table("conversations")
