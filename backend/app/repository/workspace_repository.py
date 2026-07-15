import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.rbac import Role
from app.models.user import User
from app.models.workspace import Workspace, WorkspaceMember


class WorkspaceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def add(self, workspace: Workspace) -> None:
        self.session.add(workspace)

    async def get(self, workspace_id: uuid.UUID) -> Workspace | None:
        return await self.session.get(Workspace, workspace_id)

    async def delete(self, workspace: Workspace) -> None:
        await self.session.delete(workspace)

    async def list_for_user(self, user_id: uuid.UUID) -> list[tuple[Workspace, str]]:
        stmt = (
            select(Workspace, Role.name)
            .join(WorkspaceMember, WorkspaceMember.workspace_id == Workspace.id)
            .join(Role, Role.id == WorkspaceMember.role_id)
            .where(WorkspaceMember.user_id == user_id)
            .order_by(Workspace.created_at.desc())
        )
        return [(ws, role) for ws, role in (await self.session.execute(stmt)).all()]

    async def members_with_roles(
        self, workspace_id: uuid.UUID
    ) -> list[tuple[WorkspaceMember, User, str]]:
        stmt = (
            select(WorkspaceMember, User, Role.name)
            .join(User, User.id == WorkspaceMember.user_id)
            .join(Role, Role.id == WorkspaceMember.role_id)
            .where(WorkspaceMember.workspace_id == workspace_id)
            .order_by(WorkspaceMember.joined_at)
        )
        return [
            (member, user, role_name)
            for member, user, role_name in (await self.session.execute(stmt)).all()
        ]

    async def get_member(
        self, workspace_id: uuid.UUID, user_id: uuid.UUID
    ) -> WorkspaceMember | None:
        return await self.session.get(WorkspaceMember, (workspace_id, user_id))

    def add_member(self, member: WorkspaceMember) -> None:
        self.session.add(member)

    async def delete_member(self, member: WorkspaceMember) -> None:
        await self.session.delete(member)

    async def owner_count(self, workspace_id: uuid.UUID, owner_role_name: str) -> int:
        stmt = (
            select(func.count())
            .select_from(WorkspaceMember)
            .join(Role, Role.id == WorkspaceMember.role_id)
            .where(WorkspaceMember.workspace_id == workspace_id, Role.name == owner_role_name)
        )
        return (await self.session.execute(stmt)).scalar_one()
