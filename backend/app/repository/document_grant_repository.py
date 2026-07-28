import uuid

from sqlalchemy import delete, exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_grant import DocumentGrant, PrincipalType


class DocumentGrantRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def add(self, grant: DocumentGrant) -> None:
        self.session.add(grant)

    async def get(
        self, document_id: uuid.UUID, principal_type: str, principal_id: uuid.UUID
    ) -> DocumentGrant | None:
        return await self.session.get(DocumentGrant, (document_id, principal_type, principal_id))

    async def delete(self, grant: DocumentGrant) -> None:
        await self.session.delete(grant)

    async def list_for_document(self, document_id: uuid.UUID) -> list[DocumentGrant]:
        stmt = (
            select(DocumentGrant)
            .where(DocumentGrant.document_id == document_id)
            .order_by(DocumentGrant.granted_at)
        )
        return list((await self.session.execute(stmt)).scalars())

    async def delete_for_principal(
        self, workspace_id: uuid.UUID, principal_type: PrincipalType, principal_id: uuid.UUID
    ) -> None:
        """Drop every grant to a principal within one workspace.

        ``principal_id`` is polymorphic and therefore has no foreign key, so the
        database cannot cascade this for us: without it, removing a member and
        re-adding them silently restores access to documents shared with them.
        """
        stmt = delete(DocumentGrant).where(
            DocumentGrant.principal_type == principal_type,
            DocumentGrant.principal_id == principal_id,
            DocumentGrant.document_id.in_(
                select(Document.id).where(Document.workspace_id == workspace_id)
            ),
        )
        # nothing in this unit of work reads the deleted rows again, so skip the
        # extra SELECT that session synchronisation would otherwise issue
        await self.session.execute(stmt, execution_options={"synchronize_session": False})

    async def user_has_grant(self, document_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        stmt = select(
            exists().where(
                DocumentGrant.document_id == document_id,
                DocumentGrant.principal_type == PrincipalType.USER,
                DocumentGrant.principal_id == user_id,
            )
        )
        return bool((await self.session.execute(stmt)).scalar())

    async def team_grant_exists(self, document_id: uuid.UUID, team_ids: list[uuid.UUID]) -> bool:
        if not team_ids:
            return False
        stmt = select(
            exists().where(
                DocumentGrant.document_id == document_id,
                DocumentGrant.principal_type == PrincipalType.TEAM,
                DocumentGrant.principal_id.in_(team_ids),
            )
        )
        return bool((await self.session.execute(stmt)).scalar())
