import uuid

from app.controller.team_controller.dto.team_dto import (
    AddTeamMemberRequest,
    TeamCreate,
    TeamMemberOut,
    TeamOut,
)
from app.models.user import User
from app.services.team_service import TeamService


async def create_team(
    workspace_id: uuid.UUID, data: TeamCreate, user: User, service: TeamService
) -> TeamOut:
    return TeamOut.model_validate(await service.create(user, workspace_id, data))


async def list_teams(workspace_id: uuid.UUID, service: TeamService) -> list[TeamOut]:
    teams = await service.list_for_workspace(workspace_id)
    return [TeamOut.model_validate(team) for team in teams]


async def delete_team(
    workspace_id: uuid.UUID, team_id: uuid.UUID, user: User, service: TeamService
) -> None:
    await service.delete(user, workspace_id, team_id)


async def list_team_members(
    workspace_id: uuid.UUID, team_id: uuid.UUID, service: TeamService
) -> list[TeamMemberOut]:
    return await service.members(workspace_id, team_id)


async def add_team_member(
    workspace_id: uuid.UUID,
    team_id: uuid.UUID,
    data: AddTeamMemberRequest,
    user: User,
    service: TeamService,
) -> None:
    await service.add_member(user, workspace_id, team_id, data.user_id)


async def remove_team_member(
    workspace_id: uuid.UUID,
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    user: User,
    service: TeamService,
) -> None:
    await service.remove_member(user, workspace_id, team_id, user_id)
