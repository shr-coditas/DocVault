import uuid

import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, NotFoundError
from app.modules.audit.service import AuditService
from app.modules.folders.models import Folder
from app.modules.folders.schemas import FolderCreate, FolderTreeItem, FolderUpdate
from app.modules.users.models import User


class FolderService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.audit = AuditService(session)

    async def create(self, actor: User, workspace_id: uuid.UUID, data: FolderCreate) -> Folder:
        if data.parent_id is not None:
            await self._get(workspace_id, data.parent_id)  # 404 if missing/foreign
        await self._ensure_name_free(workspace_id, data.parent_id, data.name)

        folder = Folder(
            workspace_id=workspace_id,
            parent_id=data.parent_id,
            name=data.name,
            created_by=actor.id,
        )
        self.session.add(folder)
        await self.session.flush()
        self.audit.record(
            action="folder.created",
            resource_type="folder",
            resource_id=folder.id,
            workspace_id=workspace_id,
            actor_id=actor.id,
            name=folder.name,
        )
        await self.session.commit()
        await self.session.refresh(folder)
        return folder

    async def children(self, workspace_id: uuid.UUID, parent_id: uuid.UUID | None) -> list[Folder]:
        stmt = (
            select(Folder)
            .where(Folder.workspace_id == workspace_id, Folder.parent_id == parent_id)
            .order_by(Folder.name)
        )
        return list((await self.session.execute(stmt)).scalars())

    async def tree(self, workspace_id: uuid.UUID) -> list[FolderTreeItem]:
        """Whole workspace folder tree in one query via a recursive CTE."""
        base = (
            select(
                Folder.id,
                Folder.parent_id,
                Folder.name,
                sa.literal(0).label("depth"),
                Folder.name.concat(sa.literal("")).label("path"),
            )
            .where(Folder.workspace_id == workspace_id, Folder.parent_id.is_(None))
            .cte("folder_tree", recursive=True)
        )
        child = sa.orm.aliased(Folder)
        recursive = select(
            child.id,
            child.parent_id,
            child.name,
            (base.c.depth + 1).label("depth"),
            base.c.path.concat("/").concat(child.name).label("path"),
        ).join(base, child.parent_id == base.c.id)
        tree = base.union_all(recursive)

        rows = (await self.session.execute(select(tree).order_by(tree.c.path))).all()
        return [
            FolderTreeItem(id=r.id, parent_id=r.parent_id, name=r.name, depth=r.depth, path=r.path)
            for r in rows
        ]

    async def update(
        self, actor: User, workspace_id: uuid.UUID, folder_id: uuid.UUID, data: FolderUpdate
    ) -> Folder:
        folder = await self._get(workspace_id, folder_id)
        changes = data.model_dump(exclude_unset=True)
        if not changes:
            return folder

        new_parent = changes.get("parent_id", folder.parent_id)
        new_name = changes.get("name") or folder.name

        if "parent_id" in changes and new_parent != folder.parent_id:
            if new_parent is not None:
                await self._get(workspace_id, new_parent)
                if new_parent == folder.id or await self._is_descendant(folder.id, new_parent):
                    raise ConflictError("cannot move a folder into its own subtree")
            folder.parent_id = new_parent

        if new_name != folder.name or "parent_id" in changes:
            await self._ensure_name_free(workspace_id, new_parent, new_name, exclude=folder.id)
        folder.name = new_name

        self.audit.record(
            action="folder.updated",
            resource_type="folder",
            resource_id=folder.id,
            workspace_id=workspace_id,
            actor_id=actor.id,
            **{k: str(v) for k, v in changes.items()},
        )
        await self.session.commit()
        await self.session.refresh(folder)
        return folder

    async def delete(self, actor: User, workspace_id: uuid.UUID, folder_id: uuid.UUID) -> None:
        folder = await self._get(workspace_id, folder_id)
        self.audit.record(
            action="folder.deleted",
            resource_type="folder",
            resource_id=folder.id,
            workspace_id=workspace_id,
            actor_id=actor.id,
            name=folder.name,
        )
        await self.session.delete(folder)  # DB cascades the subtree
        await self.session.commit()

    async def _get(self, workspace_id: uuid.UUID, folder_id: uuid.UUID) -> Folder:
        folder = await self.session.get(Folder, folder_id)
        if folder is None or folder.workspace_id != workspace_id:
            raise NotFoundError("folder not found")
        return folder

    async def _ensure_name_free(
        self,
        workspace_id: uuid.UUID,
        parent_id: uuid.UUID | None,
        name: str,
        exclude: uuid.UUID | None = None,
    ) -> None:
        stmt = select(Folder.id).where(
            Folder.workspace_id == workspace_id,
            Folder.parent_id == parent_id,
            Folder.name == name,
        )
        if exclude is not None:
            stmt = stmt.where(Folder.id != exclude)
        if (await self.session.execute(stmt.limit(1))).scalar_one_or_none() is not None:
            raise ConflictError("a folder with this name already exists here")

    async def _is_descendant(self, ancestor_id: uuid.UUID, candidate_id: uuid.UUID) -> bool:
        base = (
            select(Folder.id)
            .where(Folder.parent_id == ancestor_id)
            .cte("descendants", recursive=True)
        )
        child = sa.orm.aliased(Folder)
        descendants = base.union_all(select(child.id).join(base, child.parent_id == base.c.id))
        stmt = select(descendants.c.id).where(descendants.c.id == candidate_id).limit(1)
        return (await self.session.execute(stmt)).scalar_one_or_none() is not None
