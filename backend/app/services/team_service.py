import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.controller.team_controller.dto.team_dto import TeamCreate, TeamMemberOut
from app.exceptions import ConflictError, NotFoundError
from app.models.team import Team, TeamMember
from app.models.user import User
from app.repository.team_repository import TeamRepository
from app.repository.workspace_repository import WorkspaceRepository
from app.services.audit_service import AuditService


class TeamService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = TeamRepository(session)
        self.workspaces = WorkspaceRepository(session)
        self.audit = AuditService(session)

    async def create(self, actor: User, workspace_id: uuid.UUID, data: TeamCreate) -> Team:
        if await self.repository.get_by_name(workspace_id, data.name) is not None:
            raise ConflictError("a team with this name already exists in the workspace")

        team = Team(workspace_id=workspace_id, name=data.name)
        self.repository.add(team)
        await self.session.flush()
        self.audit.record(
            action="team.created",
            resource_type="team",
            resource_id=team.id,
            workspace_id=workspace_id,
            actor_id=actor.id,
            name=team.name,
        )
        await self.session.commit()
        await self.session.refresh(team)
        return team

    async def list_for_workspace(self, workspace_id: uuid.UUID) -> list[Team]:
        return await self.repository.list_for_workspace(workspace_id)

    async def delete(self, actor: User, workspace_id: uuid.UUID, team_id: uuid.UUID) -> None:
        team = await self._get_in_workspace(workspace_id, team_id)
        self.audit.record(
            action="team.deleted",
            resource_type="team",
            resource_id=team.id,
            workspace_id=workspace_id,
            actor_id=actor.id,
            name=team.name,
        )
        await self.repository.delete(team)
        await self.session.commit()

    async def add_member(
        self, actor: User, workspace_id: uuid.UUID, team_id: uuid.UUID, user_id: uuid.UUID
    ) -> None:
        await self._get_in_workspace(workspace_id, team_id)
        if await self.workspaces.get_member(workspace_id, user_id) is None:
            raise ConflictError("user is not a member of this workspace")
        if await self.repository.get_member(team_id, user_id) is not None:
            raise ConflictError("user is already in this team")

        self.repository.add_member(TeamMember(team_id=team_id, user_id=user_id))
        self.audit.record(
            action="team.member_added",
            resource_type="team",
            resource_id=team_id,
            workspace_id=workspace_id,
            actor_id=actor.id,
            member_id=str(user_id),
        )
        await self.session.commit()

    async def remove_member(
        self, actor: User, workspace_id: uuid.UUID, team_id: uuid.UUID, user_id: uuid.UUID
    ) -> None:
        await self._get_in_workspace(workspace_id, team_id)
        member = await self.repository.get_member(team_id, user_id)
        if member is None:
            raise NotFoundError("user is not in this team")

        await self.repository.delete_member(member)
        self.audit.record(
            action="team.member_removed",
            resource_type="team",
            resource_id=team_id,
            workspace_id=workspace_id,
            actor_id=actor.id,
            member_id=str(user_id),
        )
        await self.session.commit()

    async def members(self, workspace_id: uuid.UUID, team_id: uuid.UUID) -> list[TeamMemberOut]:
        await self._get_in_workspace(workspace_id, team_id)
        return [
            TeamMemberOut(user_id=u.id, email=u.email, full_name=u.full_name)
            for u in await self.repository.member_users(team_id)
        ]

    async def _get_in_workspace(self, workspace_id: uuid.UUID, team_id: uuid.UUID) -> Team:
        team = await self.repository.get(team_id)
        if team is None or team.workspace_id != workspace_id:
            raise NotFoundError("team not found")
        return team
