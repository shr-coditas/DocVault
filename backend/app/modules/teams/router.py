import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.core.deps import DbSession
from app.modules.rbac.catalog import Perm
from app.modules.rbac.deps import require_permission
from app.modules.teams.schemas import (
    AddTeamMemberRequest,
    TeamCreate,
    TeamMemberOut,
    TeamOut,
)
from app.modules.teams.service import TeamService
from app.modules.users.models import User

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
    return TeamOut.model_validate(await service.create(user, workspace_id, data))


@router.get("")
async def list_teams(workspace_id: uuid.UUID, user: CanRead, service: ServiceDep) -> list[TeamOut]:
    teams = await service.list_for_workspace(workspace_id)
    return [TeamOut.model_validate(team) for team in teams]


@router.delete("/{team_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_team(
    workspace_id: uuid.UUID, team_id: uuid.UUID, user: CanManage, service: ServiceDep
) -> None:
    await service.delete(user, workspace_id, team_id)


@router.get("/{team_id}/members")
async def list_team_members(
    workspace_id: uuid.UUID, team_id: uuid.UUID, user: CanRead, service: ServiceDep
) -> list[TeamMemberOut]:
    return await service.members(workspace_id, team_id)


@router.post("/{team_id}/members", status_code=status.HTTP_204_NO_CONTENT)
async def add_team_member(
    workspace_id: uuid.UUID,
    team_id: uuid.UUID,
    data: AddTeamMemberRequest,
    user: CanManage,
    service: ServiceDep,
) -> None:
    await service.add_member(user, workspace_id, team_id, data.user_id)


@router.delete("/{team_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_team_member(
    workspace_id: uuid.UUID,
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    user: CanManage,
    service: ServiceDep,
) -> None:
    await service.remove_member(user, workspace_id, team_id, user_id)
