import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, NotFoundError
from app.modules.audit.service import AuditService
from app.modules.teams.models import Team, TeamMember
from app.modules.teams.schemas import TeamCreate, TeamMemberOut
from app.modules.users.models import User
from app.modules.workspaces.models import WorkspaceMember


class TeamService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.audit = AuditService(session)

    async def create(self, actor: User, workspace_id: uuid.UUID, data: TeamCreate) -> Team:
        duplicate = (
            await self.session.execute(
                select(Team).where(Team.workspace_id == workspace_id, Team.name == data.name)
            )
        ).scalar_one_or_none()
        if duplicate is not None:
            raise ConflictError("a team with this name already exists in the workspace")

        team = Team(workspace_id=workspace_id, name=data.name)
        self.session.add(team)
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
        stmt = select(Team).where(Team.workspace_id == workspace_id).order_by(Team.created_at)
        return list((await self.session.execute(stmt)).scalars())

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
        await self.session.delete(team)
        await self.session.commit()

    async def add_member(
        self, actor: User, workspace_id: uuid.UUID, team_id: uuid.UUID, user_id: uuid.UUID
    ) -> None:
        await self._get_in_workspace(workspace_id, team_id)
        if await self.session.get(WorkspaceMember, (workspace_id, user_id)) is None:
            raise ConflictError("user is not a member of this workspace")
        if await self.session.get(TeamMember, (team_id, user_id)) is not None:
            raise ConflictError("user is already in this team")

        self.session.add(TeamMember(team_id=team_id, user_id=user_id))
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
        member = await self.session.get(TeamMember, (team_id, user_id))
        if member is None:
            raise NotFoundError("user is not in this team")

        await self.session.delete(member)
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
        stmt = (
            select(User)
            .join(TeamMember, TeamMember.user_id == User.id)
            .where(TeamMember.team_id == team_id)
            .order_by(TeamMember.added_at)
        )
        return [
            TeamMemberOut(user_id=u.id, email=u.email, full_name=u.full_name)
            for u in (await self.session.execute(stmt)).scalars()
        ]

    async def _get_in_workspace(self, workspace_id: uuid.UUID, team_id: uuid.UUID) -> Team:
        team = await self.session.get(Team, team_id)
        if team is None or team.workspace_id != workspace_id:
            raise NotFoundError("team not found")
        return team
