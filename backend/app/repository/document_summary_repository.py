"""Persistence for one routing summary per document."""

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_summary import DocumentSummary


class DocumentSummaryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert(self, document_id: uuid.UUID, summary: str) -> None:
        statement = insert(DocumentSummary).values(
            document_id=document_id,
            summary=summary,
        )
        await self.session.execute(
            statement.on_conflict_do_update(
                index_elements=[DocumentSummary.document_id],
                set_={"summary": statement.excluded.summary},
            )
        )

    async def list_for_documents(
        self,
        document_ids: Sequence[uuid.UUID],
    ) -> list[DocumentSummary]:
        candidates = list(dict.fromkeys(document_ids))
        if not candidates:
            return []
        statement = select(DocumentSummary).where(DocumentSummary.document_id.in_(candidates))
        return list((await self.session.execute(statement)).scalars())
