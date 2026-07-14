import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.core.deps import DbSession
from app.modules.auth.deps import CurrentUser
from app.modules.rbac.catalog import Perm
from app.modules.rbac.deps import require_permission
from app.modules.users.models import User
from app.modules.workspaces.schemas import (
    AddMemberRequest,
    MemberOut,
    UpdateMemberRequest,
    WorkspaceCreate,
    WorkspaceOut,
    WorkspaceUpdate,
    WorkspaceWithRoleOut,
)
from app.modules.workspaces.service import WorkspaceService

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
    workspace = await service.create(user, data)
    return WorkspaceOut.model_validate(workspace)


@router.get("")
async def list_my_workspaces(user: CurrentUser, service: ServiceDep) -> list[WorkspaceWithRoleOut]:
    rows = await service.list_for_user(user.id)
    return [
        WorkspaceWithRoleOut(**WorkspaceOut.model_validate(ws).model_dump(), my_role=role)
        for ws, role in rows
    ]


@router.get("/{workspace_id}")
async def get_workspace(
    workspace_id: uuid.UUID, user: CanRead, service: ServiceDep
) -> WorkspaceOut:
    return WorkspaceOut.model_validate(await service.get(workspace_id))


@router.patch("/{workspace_id}")
async def update_workspace(
    workspace_id: uuid.UUID, data: WorkspaceUpdate, user: CanUpdate, service: ServiceDep
) -> WorkspaceOut:
    return WorkspaceOut.model_validate(await service.update(user, workspace_id, data))


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workspace(workspace_id: uuid.UUID, user: CanDelete, service: ServiceDep) -> None:
    await service.delete(user, workspace_id)


@router.get("/{workspace_id}/members")
async def list_members(
    workspace_id: uuid.UUID, user: CanRead, service: ServiceDep
) -> list[MemberOut]:
    return await service.members(workspace_id)


@router.post("/{workspace_id}/members", status_code=status.HTTP_201_CREATED)
async def add_member(
    workspace_id: uuid.UUID, data: AddMemberRequest, user: CanManageMembers, service: ServiceDep
) -> MemberOut:
    return await service.add_member(user, workspace_id, data)


@router.patch("/{workspace_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def update_member_role(
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    data: UpdateMemberRequest,
    user: CanManageMembers,
    service: ServiceDep,
) -> None:
    await service.update_member_role(user, workspace_id, user_id, data)


@router.delete("/{workspace_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_member(
    workspace_id: uuid.UUID, user_id: uuid.UUID, user: CanManageMembers, service: ServiceDep
) -> None:
    await service.remove_member(user, workspace_id, user_id)
