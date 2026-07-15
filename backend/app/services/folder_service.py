import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.controller.folder_controller.dto.folder_dto import (
    FolderCreate,
    FolderTreeItem,
    FolderUpdate,
)
from app.exceptions import ConflictError, NotFoundError
from app.models.folder import Folder
from app.models.user import User
from app.repository.folder_repository import FolderRepository
from app.services.audit_service import AuditService


class FolderService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = FolderRepository(session)
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
        self.repository.add(folder)
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
        return await self.repository.children(workspace_id, parent_id)

    async def tree(self, workspace_id: uuid.UUID) -> list[FolderTreeItem]:
        return [
            FolderTreeItem(id=id_, parent_id=parent_id, name=name, depth=depth, path=path)
            for id_, parent_id, name, depth, path in await self.repository.tree_rows(workspace_id)
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
                if new_parent == folder.id or await self.repository.is_descendant(
                    folder.id, new_parent
                ):
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
        await self.repository.delete(folder)  # DB cascades the subtree
        await self.session.commit()

    async def _get(self, workspace_id: uuid.UUID, folder_id: uuid.UUID) -> Folder:
        folder = await self.repository.get(folder_id)
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
        if await self.repository.sibling_named(workspace_id, parent_id, name, exclude) is not None:
            raise ConflictError("a folder with this name already exists here")
