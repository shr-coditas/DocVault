import uuid

from app.controller.folder_controller.dto.folder_dto import (
    FolderCreate,
    FolderOut,
    FolderTreeItem,
    FolderUpdate,
)
from app.models.user import User
from app.services.folder_service import FolderService


async def create_folder(
    workspace_id: uuid.UUID, data: FolderCreate, user: User, service: FolderService
) -> FolderOut:
    return FolderOut.model_validate(await service.create(user, workspace_id, data))


async def list_folders(
    workspace_id: uuid.UUID, parent_id: uuid.UUID | None, service: FolderService
) -> list[FolderOut]:
    folders = await service.children(workspace_id, parent_id)
    return [FolderOut.model_validate(folder) for folder in folders]


async def folder_tree(workspace_id: uuid.UUID, service: FolderService) -> list[FolderTreeItem]:
    return await service.tree(workspace_id)


async def update_folder(
    workspace_id: uuid.UUID,
    folder_id: uuid.UUID,
    data: FolderUpdate,
    user: User,
    service: FolderService,
) -> FolderOut:
    return FolderOut.model_validate(await service.update(user, workspace_id, folder_id, data))


async def delete_folder(
    workspace_id: uuid.UUID, folder_id: uuid.UUID, user: User, service: FolderService
) -> None:
    await service.delete(user, workspace_id, folder_id)
