import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.team import Team, TeamMember
from app.models.user import User


class TeamRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def add(self, team: Team) -> None:
        self.session.add(team)

    async def get(self, team_id: uuid.UUID) -> Team | None:
        return await self.session.get(Team, team_id)

    async def delete(self, team: Team) -> None:
        await self.session.delete(team)

    async def get_by_name(self, workspace_id: uuid.UUID, name: str) -> Team | None:
        stmt = select(Team).where(Team.workspace_id == workspace_id, Team.name == name)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list_for_workspace(self, workspace_id: uuid.UUID) -> list[Team]:
        stmt = select(Team).where(Team.workspace_id == workspace_id).order_by(Team.created_at)
        return list((await self.session.execute(stmt)).scalars())

    async def get_member(self, team_id: uuid.UUID, user_id: uuid.UUID) -> TeamMember | None:
        return await self.session.get(TeamMember, (team_id, user_id))

    def add_member(self, member: TeamMember) -> None:
        self.session.add(member)

    async def delete_member(self, member: TeamMember) -> None:
        await self.session.delete(member)

    async def member_users(self, team_id: uuid.UUID) -> list[User]:
        stmt = (
            select(User)
            .join(TeamMember, TeamMember.user_id == User.id)
            .where(TeamMember.team_id == team_id)
            .order_by(TeamMember.added_at)
        )
        return list((await self.session.execute(stmt)).scalars())

    async def team_ids_for_user(
        self, workspace_id: uuid.UUID, user_id: uuid.UUID
    ) -> list[uuid.UUID]:
        """IDs of the teams in this workspace that the user belongs to."""
        stmt = (
            select(Team.id)
            .join(TeamMember, TeamMember.team_id == Team.id)
            .where(Team.workspace_id == workspace_id, TeamMember.user_id == user_id)
        )
        return list((await self.session.execute(stmt)).scalars())
