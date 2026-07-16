import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def add(self, document: Document) -> None:
        self.session.add(document)

    async def get(self, document_id: uuid.UUID) -> Document | None:
        return await self.session.get(Document, document_id)

    async def list_in_workspace(
        self, workspace_id: uuid.UUID, folder_id: uuid.UUID | None
    ) -> list[Document]:
        stmt = (
            select(Document)
            .where(Document.workspace_id == workspace_id, Document.folder_id == folder_id)
            .order_by(Document.created_at.desc())
        )
        return list((await self.session.execute(stmt)).scalars())
