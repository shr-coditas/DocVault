import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.controller.team_controller import team_controller
from app.controller.team_controller.dto.team_dto import (
    AddTeamMemberRequest,
    TeamCreate,
    TeamMemberOut,
    TeamOut,
)
from app.dependencies import DbSession, require_permission
from app.models.user import User
from app.services.team_service import TeamService
from app.utils.rbac_catalog import Perm

router = APIRouter(prefix="/workspaces/{workspace_id}/teams", tags=["teams"])


def get_team_service(db: DbSession) -> TeamService:
    return TeamService(db)


ServiceDep = Annotated[TeamService, Depends(get_team_service)]

CanRead = Annotated[User, Depends(require_permission(Perm.TEAM_READ))]
CanManage = Annotated[User, Depends(require_permission(Perm.TEAM_MANAGE))]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_team(
    workspace_id: uuid.UUID, data: TeamCreate, user: CanManage, service: ServiceDep
) -> TeamOut:
    return await team_controller.create_team(workspace_id, data, user, service)


@router.get("")
async def list_teams(workspace_id: uuid.UUID, user: CanRead, service: ServiceDep) -> list[TeamOut]:
    return await team_controller.list_teams(workspace_id, service)


@router.delete("/{team_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_team(
    workspace_id: uuid.UUID, team_id: uuid.UUID, user: CanManage, service: ServiceDep
) -> None:
    await team_controller.delete_team(workspace_id, team_id, user, service)


@router.get("/{team_id}/members")
async def list_team_members(
    workspace_id: uuid.UUID, team_id: uuid.UUID, user: CanRead, service: ServiceDep
) -> list[TeamMemberOut]:
    return await team_controller.list_team_members(workspace_id, team_id, service)


@router.post("/{team_id}/members", status_code=status.HTTP_204_NO_CONTENT)
async def add_team_member(
    workspace_id: uuid.UUID,
    team_id: uuid.UUID,
    data: AddTeamMemberRequest,
    user: CanManage,
    service: ServiceDep,
) -> None:
    await team_controller.add_team_member(workspace_id, team_id, data, user, service)


@router.delete("/{team_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_team_member(
    workspace_id: uuid.UUID,
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    user: CanManage,
    service: ServiceDep,
) -> None:
    await team_controller.remove_team_member(workspace_id, team_id, user_id, user, service)
