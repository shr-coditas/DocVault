"""Typed retrieval chunks and their dense/lexical indexes."""

import uuid
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UUIDPrimaryKeyMixin

EMBEDDING_DIMENSIONS = 384


class DocumentChunk(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        sa.ForeignKeyConstraint(
            ["document_id", "workspace_id"],
            ["documents.id", "documents.workspace_id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("document_id", "logical_key"),
        sa.CheckConstraint("chunk_index >= 0", name="chunk_index_non_negative"),
        sa.CheckConstraint("ordinal_in_parent >= 0", name="ordinal_in_parent_non_negative"),
        sa.Index(
            "ix_document_chunks_document",
            "workspace_id",
            "document_id",
            "chunk_index",
        ),
        sa.Index("ix_document_chunks_parent_ordinal", "parent_node_id", "ordinal_in_parent"),
        sa.Index("ix_document_chunks_workspace_type", "workspace_id", "chunk_type"),
        sa.Index("ix_document_chunks_embedding_profile", "embedding_profile_id"),
        sa.Index("ix_document_chunks_search_vector", "search_vector", postgresql_using="gin"),
        sa.Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
        ),
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("document_index_runs.id", ondelete="CASCADE")
    )
    document_id: Mapped[uuid.UUID]
    workspace_id: Mapped[uuid.UUID]
    logical_key: Mapped[str] = mapped_column(sa.String(1024))
    chunk_index: Mapped[int]
    structural_node_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("document_nodes.id", ondelete="CASCADE")
    )
    parent_node_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.ForeignKey("document_nodes.id", ondelete="SET NULL")
    )
    ordinal_in_parent: Mapped[int]
    chunk_type: Mapped[str] = mapped_column(sa.String(40))
    heading_path: Mapped[list[str]] = mapped_column(ARRAY(sa.Text), default=list)
    breadcrumb: Mapped[str | None] = mapped_column(sa.Text)
    content: Mapped[str] = mapped_column(sa.Text)
    embedding_text: Mapped[str] = mapped_column(sa.Text)
    lexical_text: Mapped[str] = mapped_column(sa.Text)
    search_vector: Mapped[Any] = mapped_column(
        TSVECTOR,
        sa.Computed("to_tsvector('simple', lexical_text)", persisted=True),
    )
    embedding_token_count: Mapped[int]
    page_start: Mapped[int | None]
    page_end: Mapped[int | None]
    source_spans: Mapped[list[dict[str, object]]] = mapped_column(JSONB)
    language: Mapped[str] = mapped_column(sa.String(20), default="und")
    content_hash: Mapped[str] = mapped_column(sa.String(64))
    chunk_metadata: Mapped[dict[str, object]] = mapped_column("metadata", JSONB, default=dict)
    embedding_profile_id: Mapped[str] = mapped_column(sa.String(255))
    embedding_model: Mapped[str] = mapped_column(sa.String(100))
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now()
    )
