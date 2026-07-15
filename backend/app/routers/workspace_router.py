import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.controller.workspace_controller import workspace_controller
from app.controller.workspace_controller.dto.workspace_dto import (
    AddMemberRequest,
    MemberOut,
    UpdateMemberRequest,
    WorkspaceCreate,
    WorkspaceOut,
    WorkspaceUpdate,
    WorkspaceWithRoleOut,
)
from app.dependencies import CurrentUser, DbSession, require_permission
from app.models.user import User
from app.services.workspace_service import WorkspaceService
from app.utils.rbac_catalog import Perm

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


def get_workspace_service(db: DbSession) -> WorkspaceService:
    return WorkspaceService(db)


ServiceDep = Annotated[WorkspaceService, Depends(get_workspace_service)]

CanRead = Annotated[User, Depends(require_permission(Perm.WORKSPACE_READ))]
CanUpdate = Annotated[User, Depends(require_permission(Perm.WORKSPACE_UPDATE))]
CanDelete = Annotated[User, Depends(require_permission(Perm.WORKSPACE_DELETE))]
CanManageMembers = Annotated[User, Depends(require_permission(Perm.WORKSPACE_MANAGE_MEMBERS))]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_workspace(
    data: WorkspaceCreate, user: CurrentUser, service: ServiceDep
) -> WorkspaceOut:
    return await workspace_controller.create_workspace(data, user, service)


@router.get("")
async def list_my_workspaces(user: CurrentUser, service: ServiceDep) -> list[WorkspaceWithRoleOut]:
    return await workspace_controller.list_my_workspaces(user, service)


@router.get("/{workspace_id}")
async def get_workspace(
    workspace_id: uuid.UUID, user: CanRead, service: ServiceDep
) -> WorkspaceOut:
    return await workspace_controller.get_workspace(workspace_id, service)


@router.patch("/{workspace_id}")
async def update_workspace(
    workspace_id: uuid.UUID, data: WorkspaceUpdate, user: CanUpdate, service: ServiceDep
) -> WorkspaceOut:
    return await workspace_controller.update_workspace(workspace_id, data, user, service)


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workspace(workspace_id: uuid.UUID, user: CanDelete, service: ServiceDep) -> None:
    await workspace_controller.delete_workspace(workspace_id, user, service)


@router.get("/{workspace_id}/members")
async def list_members(
    workspace_id: uuid.UUID, user: CanRead, service: ServiceDep
) -> list[MemberOut]:
    return await workspace_controller.list_members(workspace_id, service)


@router.post("/{workspace_id}/members", status_code=status.HTTP_201_CREATED)
async def add_member(
    workspace_id: uuid.UUID, data: AddMemberRequest, user: CanManageMembers, service: ServiceDep
) -> MemberOut:
    return await workspace_controller.add_member(workspace_id, data, user, service)


@router.patch("/{workspace_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def update_member_role(
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    data: UpdateMemberRequest,
    user: CanManageMembers,
    service: ServiceDep,
) -> None:
    await workspace_controller.update_member_role(workspace_id, user_id, data, user, service)


@router.delete("/{workspace_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_member(
    workspace_id: uuid.UUID, user_id: uuid.UUID, user: CanManageMembers, service: ServiceDep
) -> None:
    await workspace_controller.remove_member(workspace_id, user_id, user, service)
