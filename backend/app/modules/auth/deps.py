import uuid
from typing import Annotated

import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings
from app.core.deps import DbSession
from app.core.exceptions import UnauthorizedError
from app.core.security import decode_access_token
from app.modules.users.models import User
from app.modules.users.repository import UserRepository

_bearer = HTTPBearer(auto_error=False)


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
    except jwt.PyJWTError:
        raise _unauthorized() from None

    user = await UserRepository(db).get(uuid.UUID(payload["sub"]))
    if user is None or not user.is_active:
        raise _unauthorized()
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
