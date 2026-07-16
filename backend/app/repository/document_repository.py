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

    async def delete(self, document: Document) -> None:
        await self.session.delete(document)

    async def list_in_workspace(
        self, workspace_id: uuid.UUID, folder_id: uuid.UUID | None
    ) -> list[Document]:
        stmt = (
            select(Document)
            .where(
                Document.workspace_id == workspace_id,
                Document.folder_id == folder_id,
                Document.deleted_at.is_(None),
            )
            .order_by(Document.created_at.desc())
        )
        return list((await self.session.execute(stmt)).scalars())

    async def list_trashed(self, workspace_id: uuid.UUID) -> list[Document]:
        stmt = (
            select(Document)
            .where(Document.workspace_id == workspace_id, Document.deleted_at.is_not(None))
            .order_by(Document.deleted_at.desc())
        )
        return list((await self.session.execute(stmt)).scalars())

    async def storage_keys_in_folders(self, folder_ids: list[uuid.UUID]) -> list[str]:
        """Storage keys of every document (active or trashed) in the given folders."""
        if not folder_ids:
            return []
        stmt = select(Document.storage_key).where(Document.folder_id.in_(folder_ids))
        return list((await self.session.execute(stmt)).scalars())
