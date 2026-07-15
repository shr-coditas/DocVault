import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.rbac import Permission, Role, role_permissions
from app.models.workspace import WorkspaceMember


class RbacRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def role_by_name(self, name: str) -> Role | None:
        return (
            await self.session.execute(select(Role).where(Role.name == name))
        ).scalar_one_or_none()

    async def get_role(self, role_id: uuid.UUID) -> Role | None:
        return await self.session.get(Role, role_id)

    async def workspace_role_name(self, user_id: uuid.UUID, workspace_id: uuid.UUID) -> str | None:
        stmt = (
            select(Role.name)
            .join(WorkspaceMember, WorkspaceMember.role_id == Role.id)
            .where(
                WorkspaceMember.user_id == user_id,
                WorkspaceMember.workspace_id == workspace_id,
            )
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def role_has_permission(self, role_name: str, perm_code: str) -> bool:
        stmt = (
            select(Permission.id)
            .join(role_permissions, role_permissions.c.permission_id == Permission.id)
            .join(Role, Role.id == role_permissions.c.role_id)
            .where(Role.name == role_name, Permission.code == perm_code)
            .limit(1)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none() is not None
