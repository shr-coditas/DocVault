"""Dense and lexical SQL for document chunks."""

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import Row, delete, func, insert, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.repository.document_repository import DocumentRepository

ITERATIVE_SCAN_MODES = frozenset({"off", "strict_order", "relaxed_order"})


class DocumentChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def delete_for_document(self, document_id: uuid.UUID) -> None:
        await self.session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id),
            execution_options={"synchronize_session": False},
        )

    async def add_many(self, rows: list[dict[str, Any]]) -> None:
        if rows:
            await self.session.execute(insert(DocumentChunk), rows)

    async def count_for_document(self, document_id: uuid.UUID) -> int:
        stmt = (
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
        )
        return (await self.session.execute(stmt)).scalar_one()

    def _accessible_stmt(
        self,
        *,
        workspace_id: uuid.UUID,
        access: tuple[uuid.UUID, list[uuid.UUID]] | None,
        document_ids: Sequence[uuid.UUID] | None,
    ) -> Select[Any]:
        stmt = (
            select(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                DocumentChunk.workspace_id == workspace_id,
                Document.workspace_id == workspace_id,
                Document.deleted_at.is_(None),
                Document.indexed.is_(True),
            )
        )
        if access is not None:
            stmt = stmt.where(DocumentRepository._accessible_condition(*access))
        if document_ids:
            stmt = stmt.where(DocumentChunk.document_id.in_(list(document_ids)))
        return stmt

    async def search_semantic(
        self,
        *,
        workspace_id: uuid.UUID,
        query_vector: Sequence[float],
        limit: int,
        semantic_min_score: float | None,
        access: tuple[uuid.UUID, list[uuid.UUID]] | None,
        document_id: uuid.UUID | None = None,
        document_ids: Sequence[uuid.UUID] | None = None,
        iterative_scan: str | None = None,
    ) -> list[Row[tuple[DocumentChunk, Document, float]]]:
        if iterative_scan is not None:
            if iterative_scan not in ITERATIVE_SCAN_MODES:
                raise ValueError(
                    f"unsupported hnsw.iterative_scan value {iterative_scan!r}; "
                    f"expected one of {sorted(ITERATIVE_SCAN_MODES)}"
                )
            await self.session.execute(text(f"SET LOCAL hnsw.iterative_scan = '{iterative_scan}'"))

        scope = [document_id] if document_id is not None else document_ids
        distance = DocumentChunk.embedding.cosine_distance(list(query_vector))
        score = (1 - distance).label("score")
        base = self._accessible_stmt(
            workspace_id=workspace_id, access=access, document_ids=scope
        ).with_only_columns(DocumentChunk, Document, score)
        if semantic_min_score is not None:
            base = base.where(distance <= 1 - semantic_min_score)
        stmt = base.order_by(distance.asc()).limit(limit)
        return list((await self.session.execute(stmt)).all())

    async def search_lexical(
        self,
        *,
        workspace_id: uuid.UUID,
        query: str,
        limit: int,
        access: tuple[uuid.UUID, list[uuid.UUID]] | None,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[Row[tuple[DocumentChunk, Document, float]]]:
        query_expression = func.websearch_to_tsquery("english", query)
        rank = func.ts_rank_cd(DocumentChunk.search_vector, query_expression).label("score")
        stmt = (
            self._accessible_stmt(
                workspace_id=workspace_id, access=access, document_ids=document_ids
            )
            .with_only_columns(DocumentChunk, Document, rank)
            .where(DocumentChunk.search_vector.op("@@")(query_expression))
            .order_by(rank.desc())
            .limit(limit)
        )
        return list((await self.session.execute(stmt)).all())
