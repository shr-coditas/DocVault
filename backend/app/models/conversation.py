"""Persistent, creator-owned document conversations and source provenance."""

import uuid
from datetime import datetime
from enum import StrEnum

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class ConversationScope(StrEnum):
    WORKSPACE = "workspace"
    SELECTED = "selected"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class MessageStatus(StrEnum):
    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"


class MessageKind(StrEnum):
    ANSWER = "answer"
    NO_SOURCES = "no_sources"
    GENERATION_UNAVAILABLE = "generation_unavailable"
    REFUSAL = "refusal"
    DECLINE = "decline"
    CHITCHAT = "chitchat"
    CLARIFICATION = "clarification"
    SCOPE_UNAVAILABLE = "scope_unavailable"
    ERROR = "error"
    # Added in 6E. Both describe a turn that retrieved successfully and still
    # shows no answer, and both previously had to borrow a kind that misreported
    # why: `no_sources` claims nothing was found when passages were, and
    # `generation_unavailable` blames a provider that answered fine or was never
    # called. A history nobody can read honestly is not much of a ledger.
    UNSUPPORTED_EVIDENCE = "unsupported_evidence"
    ANSWER_REJECTED = "answer_rejected"


def _sql_values(enum: type[StrEnum]) -> str:
    return ", ".join(f"'{member.value}'" for member in enum)


_SCOPES = _sql_values(ConversationScope)
_ROLES = _sql_values(MessageRole)
_STATUSES = _sql_values(MessageStatus)
_KINDS = _sql_values(MessageKind)


class Conversation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "conversations"
    __table_args__ = (
        sa.CheckConstraint(f"scope_mode in ({_SCOPES})", name="scope_mode"),
        sa.Index(
            "ix_conversations_creator_workspace_updated",
            "created_by",
            "workspace_id",
            "updated_at",
            "id",
        ),
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("workspaces.id", ondelete="CASCADE")
    )
    created_by: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("users.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(sa.String(160))
    scope_mode: Mapped[str] = mapped_column(sa.String(20))


class ConversationDocument(Base):
    """One immutable member of a selected conversation's original scope.

    ``document_id`` deliberately has no foreign key. The frozen identity and
    names survive trashing and permanent deletion so scope degradation can be
    explained without preserving the document itself.
    """

    __tablename__ = "conversation_documents"
    __table_args__ = (
        sa.UniqueConstraint(
            "conversation_id",
            "position",
            name="uq_conversation_documents_conversation_position",
        ),
        sa.CheckConstraint("position >= 0 and position < 10", name="position_range"),
    )

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("conversations.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    position: Mapped[int]
    title_snapshot: Mapped[str] = mapped_column(sa.String(255))
    file_name_snapshot: Mapped[str] = mapped_column(sa.String(255))


class ConversationMessage(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "messages"
    __table_args__ = (
        sa.UniqueConstraint(
            "conversation_id", "sequence", name="uq_messages_conversation_sequence"
        ),
        sa.UniqueConstraint(
            "conversation_id",
            "turn_id",
            "role",
            name="uq_messages_conversation_turn_role",
        ),
        sa.UniqueConstraint(
            "conversation_id",
            "client_message_id",
            name="uq_messages_conversation_client_message",
        ),
        sa.CheckConstraint("sequence > 0", name="sequence_positive"),
        sa.CheckConstraint(f"role in ({_ROLES})", name="role"),
        sa.CheckConstraint(f"status in ({_STATUSES})", name="status"),
        sa.CheckConstraint(f"kind is null or kind in ({_KINDS})", name="kind"),
        sa.CheckConstraint(
            "input_tokens is null or input_tokens >= 0", name="input_tokens_non_negative"
        ),
        sa.CheckConstraint(
            "output_tokens is null or output_tokens >= 0", name="output_tokens_non_negative"
        ),
        sa.CheckConstraint(
            "(role = 'user' and status = 'complete' and kind is null "
            "and content is not null and client_message_id is not null "
            "and lease_token is null and lease_expires_at is null) or "
            "(role = 'assistant' and client_message_id is null and "
            "((status = 'pending' and kind is null and content is null "
            "and lease_token is not null and lease_expires_at is not null) or "
            "(status in ('complete', 'failed') and kind is not null "
            "and content is not null and lease_token is null "
            "and lease_expires_at is null)))",
            name="lifecycle",
        ),
        sa.Index(
            "uq_messages_conversation_pending",
            "conversation_id",
            unique=True,
            postgresql_where=sa.text("role = 'assistant' and status = 'pending'"),
        ),
        sa.Index("ix_messages_status_lease", "status", "lease_expires_at"),
    )

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("conversations.id", ondelete="CASCADE")
    )
    turn_id: Mapped[uuid.UUID]
    sequence: Mapped[int]
    role: Mapped[str] = mapped_column(sa.String(20))
    status: Mapped[str] = mapped_column(sa.String(20))
    kind: Mapped[str | None] = mapped_column(sa.String(40))
    content: Mapped[str | None] = mapped_column(sa.Text)
    context_eligible: Mapped[bool] = mapped_column(default=False, server_default=sa.false())
    resolved_query: Mapped[str | None] = mapped_column(sa.Text)
    client_message_id: Mapped[uuid.UUID | None]
    model: Mapped[str | None] = mapped_column(sa.String(255))
    input_tokens: Mapped[int | None]
    output_tokens: Mapped[int | None]
    lease_token: Mapped[uuid.UUID | None] = mapped_column(unique=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))


class MessageSource(UUIDPrimaryKeyMixin, Base):
    """Frozen source identity for one retrieved hit.

    Document and chunk identities intentionally have no foreign keys. A source
    row records what a historical turn used even after a reindex or permanent
    document deletion. Conversation deletion still cascades through ``message``.
    """

    __tablename__ = "message_sources"
    __table_args__ = (
        sa.UniqueConstraint("message_id", "retrieval_rank", name="uq_message_sources_message_rank"),
        sa.UniqueConstraint(
            "message_id",
            "document_id",
            "index_generation",
            "logical_key",
            name="uq_message_sources_message_logical_source",
        ),
        sa.UniqueConstraint(
            "message_id", "citation_marker", name="uq_message_sources_message_citation"
        ),
        sa.CheckConstraint("retrieval_rank > 0", name="retrieval_rank_positive"),
        sa.CheckConstraint("index_generation > 0", name="index_generation_positive"),
        sa.CheckConstraint(
            "citation_marker is null or citation_marker > 0", name="citation_marker_positive"
        ),
        sa.CheckConstraint(
            "citation_marker is null or supplied_to_model", name="citation_requires_supplied"
        ),
        sa.Index("ix_message_sources_document_logical", "document_id", "logical_key"),
    )

    message_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("messages.id", ondelete="CASCADE"))
    document_id: Mapped[uuid.UUID]
    chunk_id: Mapped[uuid.UUID]
    index_generation: Mapped[int]
    logical_key: Mapped[str] = mapped_column(sa.String(1024))
    document_title_snapshot: Mapped[str] = mapped_column(sa.String(255))
    heading: Mapped[str | None] = mapped_column(sa.Text)
    breadcrumb: Mapped[str | None] = mapped_column(sa.Text)
    page_numbers: Mapped[list[int]] = mapped_column(ARRAY(sa.Integer), default=list)
    source_spans: Mapped[list[dict[str, object]]] = mapped_column(JSONB, default=list)
    retrieval_rank: Mapped[int]
    supplied_to_model: Mapped[bool] = mapped_column(default=False, server_default=sa.false())
    citation_marker: Mapped[int | None]
