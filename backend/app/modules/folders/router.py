import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from app.core.deps import DbSession
from app.modules.folders.schemas import FolderCreate, FolderOut, FolderTreeItem, FolderUpdate
from app.modules.folders.service import FolderService
from app.modules.rbac.catalog import Perm
from app.modules.rbac.deps import require_permission
from app.modules.users.models import User

router = APIRouter(prefix="/workspaces/{workspace_id}/folders", tags=["folders"])


def get_folder_service(db: DbSession) -> FolderService:
    return FolderService(db)


ServiceDep = Annotated[FolderService, Depends(get_folder_service)]

CanRead = Annotated[User, Depends(require_permission(Perm.DOCUMENT_READ))]
CanCreate = Annotated[User, Depends(require_permission(Perm.DOCUMENT_CREATE))]
CanUpdate = Annotated[User, Depends(require_permission(Perm.DOCUMENT_UPDATE))]
CanDelete = Annotated[User, Depends(require_permission(Perm.DOCUMENT_DELETE))]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_folder(
    workspace_id: uuid.UUID, data: FolderCreate, user: CanCreate, service: ServiceDep
) -> FolderOut:
    return FolderOut.model_validate(await service.create(user, workspace_id, data))


@router.get("")
async def list_folders(
    workspace_id: uuid.UUID,
    user: CanRead,
    service: ServiceDep,
    parent_id: Annotated[uuid.UUID | None, Query()] = None,
) -> list[FolderOut]:
    folders = await service.children(workspace_id, parent_id)
    return [FolderOut.model_validate(folder) for folder in folders]


@router.get("/tree")
async def folder_tree(
    workspace_id: uuid.UUID, user: CanRead, service: ServiceDep
) -> list[FolderTreeItem]:
    return await service.tree(workspace_id)


@router.patch("/{folder_id}")
async def update_folder(
    workspace_id: uuid.UUID,
    folder_id: uuid.UUID,
    data: FolderUpdate,
    user: CanUpdate,
    service: ServiceDep,
) -> FolderOut:
    return FolderOut.model_validate(await service.update(user, workspace_id, folder_id, data))


@router.delete("/{folder_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_folder(
    workspace_id: uuid.UUID, folder_id: uuid.UUID, user: CanDelete, service: ServiceDep
) -> None:
    await service.delete(user, workspace_id, folder_id)
