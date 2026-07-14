import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ForbiddenError, NotFoundError
from app.modules.rbac.catalog import Perm
from app.modules.rbac.models import Permission, Role, role_permissions
from app.modules.workspaces.models import WorkspaceMember


class PermissionService:
    """Single choke point for workspace authorization decisions."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def role_by_name(self, name: str) -> Role:
        role = (
            await self.session.execute(select(Role).where(Role.name == name))
        ).scalar_one_or_none()
        if role is None:
            raise NotFoundError(f"role '{name}' does not exist")
        return role

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

    async def require(self, user_id: uuid.UUID, workspace_id: uuid.UUID, perm: Perm) -> str:
        """Return the user's role name, or raise.

        Non-members get 404 (they should not learn the workspace exists);
        members without the permission get 403.
        """
        role_name = await self.workspace_role_name(user_id, workspace_id)
        if role_name is None:
            raise NotFoundError("workspace not found")

        stmt = (
            select(Permission.id)
            .join(role_permissions, role_permissions.c.permission_id == Permission.id)
            .join(Role, Role.id == role_permissions.c.role_id)
            .where(Role.name == role_name, Permission.code == perm.value)
            .limit(1)
        )
        allowed = (await self.session.execute(stmt)).scalar_one_or_none() is not None
        if not allowed:
            raise ForbiddenError(f"requires permission '{perm.value}'")
        return role_name
