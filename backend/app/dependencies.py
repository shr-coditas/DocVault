"""Shared FastAPI dependencies: DB session, current user, permission guards."""

import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated

import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.session import get_db
from app.exceptions import UnauthorizedError
from app.models.user import User
from app.repository.user_repository import UserRepository
from app.services.permission_service import PermissionService
from app.services.storage_service import StorageService
from app.utils.rbac_catalog import Perm
from app.utils.security import decode_access_token

DbSession = Annotated[AsyncSession, Depends(get_db)]


def get_storage_service() -> StorageService:
    """Object-storage seam: tests override this to point at a throwaway MinIO."""
    return StorageService()


StorageDep = Annotated[StorageService, Depends(get_storage_service)]

_bearer = HTTPBearer(
    auto_error=False
)  # auto_error= false means that if no credentials are provided, it will return None instead of raising an error. This allows us to handle the case where the user is not authenticated and raise a custom UnauthorizedError.


def _unauthorized() -> UnauthorizedError:
    return UnauthorizedError("not authenticated")


async def get_current_user(
    db: DbSession,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> User:
    if credentials is None:
        raise _unauthorized()

    try:
        payload = decode_access_token(credentials.credentials, get_settings().jwt_secret)
        # a correctly-signed token can still carry a missing or non-uuid `sub`;
        # that is a bad credential (401), not a server fault (500)
        user_id = uuid.UUID(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise _unauthorized() from None

    user = await UserRepository(db).get(user_id)
    if user is None or not user.is_active:
        raise _unauthorized()
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_permission(perm: Perm) -> Callable[..., Awaitable[User]]:
    """Dependency factory: guard an endpoint under /workspaces/{workspace_id}/...

    Resolves the current user, checks the permission against their workspace
    role, and returns the user for use in the handler.
    """

    async def dependency(workspace_id: uuid.UUID, user: CurrentUser, db: DbSession) -> User:
        await PermissionService(db).require(user.id, workspace_id, perm)
        return user

    return dependency
