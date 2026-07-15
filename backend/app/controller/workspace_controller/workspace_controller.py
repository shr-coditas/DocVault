import uuid

from app.controller.workspace_controller.dto.workspace_dto import (
    AddMemberRequest,
    MemberOut,
    UpdateMemberRequest,
    WorkspaceCreate,
    WorkspaceOut,
    WorkspaceUpdate,
    WorkspaceWithRoleOut,
)
from app.models.user import User
from app.services.workspace_service import WorkspaceService


async def create_workspace(
    data: WorkspaceCreate, user: User, service: WorkspaceService
) -> WorkspaceOut:
    workspace = await service.create(user, data)
    return WorkspaceOut.model_validate(workspace)


async def list_my_workspaces(user: User, service: WorkspaceService) -> list[WorkspaceWithRoleOut]:
    rows = await service.list_for_user(user.id)
    return [
        WorkspaceWithRoleOut(**WorkspaceOut.model_validate(ws).model_dump(), my_role=role)
        for ws, role in rows
    ]


async def get_workspace(workspace_id: uuid.UUID, service: WorkspaceService) -> WorkspaceOut:
    return WorkspaceOut.model_validate(await service.get(workspace_id))


async def update_workspace(
    workspace_id: uuid.UUID, data: WorkspaceUpdate, user: User, service: WorkspaceService
) -> WorkspaceOut:
    return WorkspaceOut.model_validate(await service.update(user, workspace_id, data))


async def delete_workspace(workspace_id: uuid.UUID, user: User, service: WorkspaceService) -> None:
    await service.delete(user, workspace_id)


async def list_members(workspace_id: uuid.UUID, service: WorkspaceService) -> list[MemberOut]:
    return await service.members(workspace_id)


async def add_member(
    workspace_id: uuid.UUID, data: AddMemberRequest, user: User, service: WorkspaceService
) -> MemberOut:
    return await service.add_member(user, workspace_id, data)


async def update_member_role(
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    data: UpdateMemberRequest,
    user: User,
    service: WorkspaceService,
) -> None:
    await service.update_member_role(user, workspace_id, user_id, data)


async def remove_member(
    workspace_id: uuid.UUID, user_id: uuid.UUID, user: User, service: WorkspaceService
) -> None:
    await service.remove_member(user, workspace_id, user_id)
