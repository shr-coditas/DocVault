"""Small Markdown-oriented chunks used by lexical and semantic search."""

import uuid
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR
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
        sa.UniqueConstraint("document_id", "chunk_index"),
        sa.CheckConstraint(
            "chunk_index >= 0",
            name="chunk_index_non_negative",
        ),
        sa.Index(
            "ix_document_chunks_document",
            "workspace_id",
            "document_id",
            "chunk_index",
        ),
        sa.Index(
            "ix_document_chunks_search_vector",
            "search_vector",
            postgresql_using="gin",
        ),
        sa.Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
        ),
    )

    document_id: Mapped[uuid.UUID]
    workspace_id: Mapped[uuid.UUID]
    chunk_index: Mapped[int]
    chunk_type: Mapped[str] = mapped_column(sa.String(20))
    section_path: Mapped[str | None] = mapped_column(sa.Text)
    content: Mapped[str] = mapped_column(sa.Text)
    search_vector: Mapped[Any] = mapped_column(
        TSVECTOR,
        sa.Computed(
            "to_tsvector('english', coalesce(section_path, '') || ' ' || content)",
            persisted=True,
        ),
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
    )
