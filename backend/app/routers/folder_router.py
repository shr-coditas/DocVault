import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from app.controller.folder_controller import folder_controller
from app.controller.folder_controller.dto.folder_dto import (
    FolderCreate,
    FolderOut,
    FolderTreeItem,
    FolderUpdate,
)
from app.dependencies import DbSession, StorageDep, require_permission
from app.models.user import User
from app.services.folder_service import FolderService
from app.utils.rbac_catalog import Perm

router = APIRouter(prefix="/workspaces/{workspace_id}/folders", tags=["folders"])


def get_folder_service(db: DbSession, storage: StorageDep) -> FolderService:
    return FolderService(db, storage)


ServiceDep = Annotated[FolderService, Depends(get_folder_service)]

CanRead = Annotated[User, Depends(require_permission(Perm.DOCUMENT_READ))]
CanCreate = Annotated[User, Depends(require_permission(Perm.DOCUMENT_CREATE))]
CanUpdate = Annotated[User, Depends(require_permission(Perm.DOCUMENT_UPDATE))]
CanDelete = Annotated[User, Depends(require_permission(Perm.DOCUMENT_DELETE))]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_folder(
    workspace_id: uuid.UUID, data: FolderCreate, user: CanCreate, service: ServiceDep
) -> FolderOut:
    return await folder_controller.create_folder(workspace_id, data, user, service)


@router.get("")
async def list_folders(
    workspace_id: uuid.UUID,
    user: CanRead,
    service: ServiceDep,
    parent_id: Annotated[uuid.UUID | None, Query()] = None,
) -> list[FolderOut]:
    return await folder_controller.list_folders(workspace_id, parent_id, service)


@router.get("/tree")
async def folder_tree(
    workspace_id: uuid.UUID, user: CanRead, service: ServiceDep
) -> list[FolderTreeItem]:
    return await folder_controller.folder_tree(workspace_id, service)


@router.patch("/{folder_id}")
async def update_folder(
    workspace_id: uuid.UUID,
    folder_id: uuid.UUID,
    data: FolderUpdate,
    user: CanUpdate,
    service: ServiceDep,
) -> FolderOut:
    return await folder_controller.update_folder(workspace_id, folder_id, data, user, service)


@router.delete("/{folder_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_folder(
    workspace_id: uuid.UUID, folder_id: uuid.UUID, user: CanDelete, service: ServiceDep
) -> None:
    await folder_controller.delete_folder(workspace_id, folder_id, user, service)
