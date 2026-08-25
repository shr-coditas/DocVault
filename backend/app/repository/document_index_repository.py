"""SQL for the small document-index run audit table."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_index import DocumentIndexRun


class DocumentIndexRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def add(self, run: DocumentIndexRun) -> None:
        self.session.add(run)

    async def latest_for_document(self, document_id: uuid.UUID) -> DocumentIndexRun | None:
        stmt = (
            select(DocumentIndexRun)
            .where(DocumentIndexRun.document_id == document_id)
            .order_by(DocumentIndexRun.started_at.desc(), DocumentIndexRun.id.desc())
            .limit(1)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get(self, run_id: uuid.UUID) -> DocumentIndexRun | None:
        return await self.session.get(DocumentIndexRun, run_id)
