import uuid
from collections.abc import Sequence

from sqlalchemy import ColumnElement, exists, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentVisibility
from app.models.document_grant import DocumentGrant, PrincipalType


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _accessible_condition(user_id: uuid.UUID, team_ids: list[uuid.UUID]) -> ColumnElement[bool]:
        """Rows a non-admin user may see (mirrors DocumentService._ensure_can_access).

        User and team grants are peers: both open up a restricted document
        regardless of whether its visibility is ``private`` or ``team``.
        """
        user_grant = exists().where(
            DocumentGrant.document_id == Document.id,
            DocumentGrant.principal_type == PrincipalType.USER,
            DocumentGrant.principal_id == user_id,
        )
        team_grant = exists().where(
            DocumentGrant.document_id == Document.id,
            DocumentGrant.principal_type == PrincipalType.TEAM,
            # a user in no team must match nothing; IN () is a syntax error
            DocumentGrant.principal_id.in_(team_ids or [uuid.UUID(int=0)]),
        )
        return or_(
            Document.visibility == DocumentVisibility.WORKSPACE,
            Document.owner_id == user_id,
            user_grant,
            team_grant,
        )

    def add(self, document: Document) -> None:
        self.session.add(document)

    async def get(self, document_id: uuid.UUID) -> Document | None:
        return await self.session.get(Document, document_id)

    async def delete(self, document: Document) -> None:
        await self.session.delete(document)

    async def list_in_workspace(
        self,
        workspace_id: uuid.UUID,
        folder_id: uuid.UUID | None,
        *,
        access: tuple[uuid.UUID, list[uuid.UUID]] | None = None,
        every_folder: bool = False,
    ) -> list[Document]:
        """List active documents. ``every_folder`` spans the whole workspace and
        ignores ``folder_id``; otherwise the listing is scoped to that one folder
        (``None`` = the workspace root). ``access`` = (user_id, team_ids) applies
        the visibility filter; ``None`` = admin (see everything)."""
        stmt = (
            select(Document)
            .where(
                Document.workspace_id == workspace_id,
                Document.deleted_at.is_(None),
            )
            .order_by(Document.created_at.desc())
        )
        if not every_folder:
            stmt = stmt.where(Document.folder_id == folder_id)
        if access is not None:
            stmt = stmt.where(self._accessible_condition(*access))
        return list((await self.session.execute(stmt)).scalars())

    async def list_trashed(
        self,
        workspace_id: uuid.UUID,
        *,
        access: tuple[uuid.UUID, list[uuid.UUID]] | None = None,
    ) -> list[Document]:
        stmt = (
            select(Document)
            .where(Document.workspace_id == workspace_id, Document.deleted_at.is_not(None))
            .order_by(Document.deleted_at.desc())
        )
        if access is not None:
            stmt = stmt.where(self._accessible_condition(*access))
        return list((await self.session.execute(stmt)).scalars())

    async def accessible_active_by_ids(
        self,
        workspace_id: uuid.UUID,
        document_ids: Sequence[uuid.UUID],
        *,
        access: tuple[uuid.UUID, list[uuid.UUID]] | None,
    ) -> list[Document]:
        """One batched ACL projection for scope validation and history reads.

        The caller compares the returned ids with its candidate set. Keeping the
        workspace, deletion and visibility predicates in this repository avoids
        per-document service checks and, more importantly, another copy of the
        restricted-document rule.
        """
        candidates = list(dict.fromkeys(document_ids))
        if not candidates:
            return []
        stmt = select(Document).where(
            Document.workspace_id == workspace_id,
            Document.id.in_(candidates),
            Document.deleted_at.is_(None),
        )
        if access is not None:
            stmt = stmt.where(self._accessible_condition(*access))
        return list((await self.session.execute(stmt)).scalars())

    async def unindexed_ids(
        self,
        *,
        limit: int,
        workspace_id: uuid.UUID | None = None,
        max_attempts: int | None = None,
    ) -> list[uuid.UUID]:
        """Documents the indexing script should pick up, oldest first.

        Ids only, not ORM objects: the caller gives each document its own
        transaction, so holding rows from this query open would be pointless.
        Served by the partial index ``ix_documents_unindexed``.
        """
        stmt = (
            select(Document.id)
            .where(Document.indexed.is_(False), Document.deleted_at.is_(None))
            .order_by(Document.created_at)
            .limit(limit)
        )
        if workspace_id is not None:
            stmt = stmt.where(Document.workspace_id == workspace_id)
        if max_attempts is not None:
            # stops one unparseable file consuming every future run
            stmt = stmt.where(Document.index_attempts < max_attempts)
        return list((await self.session.execute(stmt)).scalars())

    async def claim_for_indexing(self, document_id: uuid.UUID) -> Document | None:
        """Lock one document for indexing, or return None if someone else has it.

        ``SKIP LOCKED`` costs nothing for today's single-runner script and is
        exactly the primitive a competing-consumer worker pool needs, so the
        claim semantics do not have to be redesigned later.
        """
        stmt = (
            select(Document)
            .where(
                Document.id == document_id,
                Document.indexed.is_(False),
                Document.deleted_at.is_(None),
            )
            .with_for_update(skip_locked=True)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def storage_keys_in_workspace(self, workspace_id: uuid.UUID) -> list[str]:
        """Storage keys of every document (active or trashed) in a workspace."""
        stmt = select(Document.storage_key).where(Document.workspace_id == workspace_id)
        return list((await self.session.execute(stmt)).scalars())

    async def storage_keys_in_folders(self, folder_ids: list[uuid.UUID]) -> list[str]:
        """Storage keys of every document (active or trashed) in the given folders."""
        if not folder_ids:
            return []
        stmt = select(Document.storage_key).where(Document.folder_id.in_(folder_ids))
        return list((await self.session.execute(stmt)).scalars())
