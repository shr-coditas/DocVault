import uuid
from collections.abc import Awaitable, Callable

from app.core.deps import DbSession
from app.modules.auth.deps import CurrentUser
from app.modules.rbac.catalog import Perm
from app.modules.rbac.service import PermissionService
from app.modules.users.models import User


def require_permission(perm: Perm) -> Callable[..., Awaitable[User]]:
    """Dependency factory: guard an endpoint under /workspaces/{workspace_id}/...

    Resolves the current user, checks the permission against their workspace
    role, and returns the user for use in the handler.
    """

    async def dependency(workspace_id: uuid.UUID, user: CurrentUser, db: DbSession) -> User:
        await PermissionService(db).require(user.id, workspace_id, perm)
        return user

    return dependency
