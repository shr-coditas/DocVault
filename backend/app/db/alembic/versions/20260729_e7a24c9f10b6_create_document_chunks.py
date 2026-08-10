"""create structured document indexing tables

Revision ID: e7a24c9f10b6
Revises: b3f81d5a27c9
Create Date: 2026-07-29 17:45:00.000000

This uncommitted revision was deliberately rewritten before Slice 3b was
published. It is the first and only schema for structure-aware indexing.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR

revision: str = "e7a24c9f10b6"
down_revision: str | None = "b3f81d5a27c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIMENSIONS = 384


def upgrade() -> None:
    op.add_column(
        "documents", sa.Column("active_embedding_profile", sa.String(length=255), nullable=True)
    )
    op.create_unique_constraint("uq_documents_id", "documents", ["id", "workspace_id"])

    op.create_table(
        "document_index_runs",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("target_generation", sa.Integer(), nullable=False),
        sa.Column("source_checksum", sa.String(length=64), nullable=False),
        sa.Column("extractor_profile", sa.String(length=255), nullable=False),
        sa.Column("normalizer_profile", sa.String(length=255), nullable=False),
        sa.Column("chunker_profile", sa.String(length=255), nullable=False),
        sa.Column("embedding_profile", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("lease_token", sa.Uuid(), nullable=False),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("artifact_key", sa.String(length=1024), nullable=True),
        sa.Column("quality_metrics", JSONB(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id", "workspace_id"],
            ["documents.id", "documents.workspace_id"],
            name=op.f("fk_document_index_runs_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_index_runs")),
        sa.UniqueConstraint(
            "document_id",
            "target_generation",
            name=op.f("uq_document_index_runs_document_id"),
        ),
        sa.UniqueConstraint("lease_token", name=op.f("uq_document_index_runs_lease_token")),
    )
    op.create_check_constraint(
        "status",
        "document_index_runs",
        "status in ('claimed', 'extracting', 'chunking', 'embedding', 'staged', 'active', 'failed')",
    )
    op.create_check_constraint(
        "target_generation_positive", "document_index_runs", "target_generation > 0"
    )
    op.create_index(
        "ix_document_index_runs_status_lease",
        "document_index_runs",
        ["status", "lease_expires_at"],
    )

    op.create_table(
        "document_nodes",
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("index_generation", sa.Integer(), nullable=False),
        sa.Column("parent_id", sa.Uuid(), nullable=True),
        sa.Column("logical_path", sa.String(length=1024), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("node_type", sa.String(length=40), nullable=False),
        sa.Column("heading_level", sa.Integer(), nullable=True),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("source_spans", JSONB(), nullable=False),
        sa.Column("attributes", JSONB(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id", "workspace_id"],
            ["documents.id", "documents.workspace_id"],
            name=op.f("fk_document_nodes_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            ["document_nodes.id"],
            name=op.f("fk_document_nodes_parent_id_document_nodes"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["document_index_runs.id"],
            name=op.f("fk_document_nodes_run_id_document_index_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_nodes")),
        sa.UniqueConstraint("run_id", "logical_path", name=op.f("uq_document_nodes_run_id")),
    )
    op.create_check_constraint("ordinal_non_negative", "document_nodes", "ordinal >= 0")
    op.create_check_constraint(
        "confidence_range", "document_nodes", "confidence >= 0 and confidence <= 1"
    )
    op.create_index("ix_document_nodes_parent_ordinal", "document_nodes", ["parent_id", "ordinal"])
    op.create_index(
        "ix_document_nodes_document_generation",
        "document_nodes",
        ["workspace_id", "document_id", "index_generation"],
    )

    op.create_table(
        "document_chunks",
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("index_generation", sa.Integer(), nullable=False),
        sa.Column("logical_key", sa.String(length=1024), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("structural_node_id", sa.Uuid(), nullable=False),
        sa.Column("parent_node_id", sa.Uuid(), nullable=True),
        sa.Column("ordinal_in_parent", sa.Integer(), nullable=False),
        sa.Column("chunk_type", sa.String(length=40), nullable=False),
        sa.Column("heading_path", ARRAY(sa.Text()), nullable=False),
        sa.Column("breadcrumb", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding_text", sa.Text(), nullable=False),
        sa.Column("lexical_text", sa.Text(), nullable=False),
        sa.Column(
            "search_vector",
            TSVECTOR(),
            sa.Computed("to_tsvector('simple', lexical_text)", persisted=True),
            nullable=False,
        ),
        sa.Column("embedding_token_count", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("source_spans", JSONB(), nullable=False),
        sa.Column("language", sa.String(length=20), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("metadata", JSONB(), nullable=False),
        sa.Column("embedding_profile_id", sa.String(length=255), nullable=False),
        sa.Column("embedding_model", sa.String(length=100), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSIONS), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id", "workspace_id"],
            ["documents.id", "documents.workspace_id"],
            name=op.f("fk_document_chunks_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["parent_node_id"],
            ["document_nodes.id"],
            name=op.f("fk_document_chunks_parent_node_id_document_nodes"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["document_index_runs.id"],
            name=op.f("fk_document_chunks_run_id_document_index_runs"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["structural_node_id"],
            ["document_nodes.id"],
            name=op.f("fk_document_chunks_structural_node_id_document_nodes"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_chunks")),
        sa.UniqueConstraint(
            "document_id",
            "index_generation",
            "logical_key",
            name=op.f("uq_document_chunks_document_id"),
        ),
    )
    op.create_check_constraint("chunk_index_non_negative", "document_chunks", "chunk_index >= 0")
    op.create_check_constraint(
        "ordinal_in_parent_non_negative", "document_chunks", "ordinal_in_parent >= 0"
    )
    op.create_index(
        "ix_document_chunks_document_generation",
        "document_chunks",
        ["workspace_id", "document_id", "index_generation", "chunk_index"],
    )
    op.create_index(
        "ix_document_chunks_parent_ordinal",
        "document_chunks",
        ["parent_node_id", "ordinal_in_parent"],
    )
    op.create_index(
        "ix_document_chunks_workspace_type", "document_chunks", ["workspace_id", "chunk_type"]
    )
    op.create_index(
        "ix_document_chunks_embedding_profile", "document_chunks", ["embedding_profile_id"]
    )
    op.create_index(
        "ix_document_chunks_search_vector",
        "document_chunks",
        ["search_vector"],
        postgresql_using="gin",
    )
    op.execute(
        "CREATE INDEX ix_document_chunks_embedding_hnsw ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.drop_table("document_chunks")
    op.drop_table("document_nodes")
    op.drop_table("document_index_runs")
    op.drop_constraint("uq_documents_id", "documents", type_="unique")
    op.drop_column("documents", "active_embedding_profile")
