"""SQL for durable index runs and canonical document nodes."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, func, insert, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_index import DocumentIndexRun, DocumentStructureNode, IndexRunStatus


class DocumentIndexRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def add_run(self, run: DocumentIndexRun) -> None:
        self.session.add(run)

    async def run_for_target(
        self, document_id: uuid.UUID, target_generation: int
    ) -> DocumentIndexRun | None:
        stmt = select(DocumentIndexRun).where(
            DocumentIndexRun.document_id == document_id,
            DocumentIndexRun.target_generation == target_generation,
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def competing_run(
        self, document_id: uuid.UUID, *, now: datetime | None = None
    ) -> DocumentIndexRun | None:
        current = now or datetime.now(UTC)
        stmt = (
            select(DocumentIndexRun)
            .where(
                DocumentIndexRun.document_id == document_id,
                DocumentIndexRun.status.not_in(
                    [IndexRunStatus.ACTIVE.value, IndexRunStatus.FAILED.value]
                ),
                or_(
                    DocumentIndexRun.lease_expires_at > current,
                    DocumentIndexRun.status == IndexRunStatus.STAGED.value,
                ),
            )
            .order_by(DocumentIndexRun.created_at.desc())
            .limit(1)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get(self, run_id: uuid.UUID) -> DocumentIndexRun | None:
        return await self.session.get(DocumentIndexRun, run_id)

    async def add_nodes(self, rows: list[dict[str, Any]]) -> None:
        if rows:
            await self.session.execute(insert(DocumentStructureNode), rows)

    async def delete_staged(self, run_id: uuid.UUID) -> None:
        await self.session.execute(
            delete(DocumentStructureNode).where(DocumentStructureNode.run_id == run_id)
        )

    async def node_count(self, run_id: uuid.UUID) -> int:
        stmt = (
            select(func.count())
            .select_from(DocumentStructureNode)
            .where(DocumentStructureNode.run_id == run_id)
        )
        return (await self.session.execute(stmt)).scalar_one()

    async def invalid_node_count(self, run_id: uuid.UUID) -> int:
        stmt = (
            select(func.count())
            .select_from(DocumentStructureNode)
            .where(
                DocumentStructureNode.run_id == run_id,
                or_(
                    DocumentStructureNode.content_hash.is_(None),
                    func.length(DocumentStructureNode.content_hash) != 64,
                ),
            )
        )
        return (await self.session.execute(stmt)).scalar_one()

    async def delete_generations_older_than(
        self, document_id: uuid.UUID, minimum_generation: int
    ) -> None:
        stale_runs = select(DocumentIndexRun.id).where(
            DocumentIndexRun.document_id == document_id,
            DocumentIndexRun.target_generation < minimum_generation,
        )
        await self.session.execute(
            delete(DocumentIndexRun).where(DocumentIndexRun.id.in_(stale_runs))
        )

    async def expired_nonterminal(self, now: datetime) -> list[DocumentIndexRun]:
        stmt = select(DocumentIndexRun).where(
            DocumentIndexRun.status.not_in(
                [IndexRunStatus.ACTIVE.value, IndexRunStatus.FAILED.value]
            ),
            DocumentIndexRun.lease_expires_at <= now,
        )
        return list((await self.session.execute(stmt)).scalars())

    async def artifacts_before(self, cutoff: datetime) -> list[DocumentIndexRun]:
        stmt = select(DocumentIndexRun).where(
            DocumentIndexRun.artifact_key.is_not(None),
            DocumentIndexRun.created_at < cutoff,
            DocumentIndexRun.status.in_([IndexRunStatus.ACTIVE.value, IndexRunStatus.FAILED.value]),
        )
        return list((await self.session.execute(stmt)).scalars())

    async def nodes_by_ids(
        self, ids: list[uuid.UUID], *, document_id: uuid.UUID, generation: int
    ) -> list[DocumentStructureNode]:
        if not ids:
            return []
        stmt = select(DocumentStructureNode).where(
            DocumentStructureNode.id.in_(ids),
            DocumentStructureNode.document_id == document_id,
            DocumentStructureNode.index_generation == generation,
        )
        return list((await self.session.execute(stmt)).scalars())
