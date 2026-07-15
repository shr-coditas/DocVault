import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions import ForbiddenError, NotFoundError
from app.models.rbac import Role
from app.repository.rbac_repository import RbacRepository
from app.utils.rbac_catalog import Perm


class PermissionService:
    """Single choke point for workspace authorization decisions."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = RbacRepository(session)

    async def role_by_name(self, name: str) -> Role:
        role = await self.repository.role_by_name(name)
        if role is None:
            raise NotFoundError(f"role '{name}' does not exist")
        return role

    async def workspace_role_name(self, user_id: uuid.UUID, workspace_id: uuid.UUID) -> str | None:
        return await self.repository.workspace_role_name(user_id, workspace_id)

    async def require(self, user_id: uuid.UUID, workspace_id: uuid.UUID, perm: Perm) -> str:
        """Return the user's role name, or raise.

        Non-members get 404 (they should not learn the workspace exists);
        members without the permission get 403.
        """
        role_name = await self.repository.workspace_role_name(user_id, workspace_id)
        if role_name is None:
            raise NotFoundError("workspace not found")

        if not await self.repository.role_has_permission(role_name, perm.value):
            raise ForbiddenError(f"requires permission '{perm.value}'")
        return role_name
