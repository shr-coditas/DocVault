from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.core.deps import DbSession
from app.modules.auth.deps import CurrentUser
from app.modules.auth.schemas import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
)
from app.modules.auth.service import AuthService
from app.modules.users.schemas import UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


def get_auth_service(db: DbSession) -> AuthService:
    return AuthService(db)


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(data: RegisterRequest, service: AuthServiceDep) -> UserOut:
    user = await service.register(data)
    return UserOut.model_validate(user)


@router.post("/login")
async def login(data: LoginRequest, service: AuthServiceDep) -> TokenResponse:
    return await service.login(data)


@router.post("/refresh")
async def refresh(data: RefreshRequest, service: AuthServiceDep) -> TokenResponse:
    return await service.refresh(data.refresh_token)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(data: RefreshRequest, service: AuthServiceDep) -> None:
    await service.logout(data.refresh_token)


@router.get("/me")
async def me(user: CurrentUser) -> UserOut:
    return UserOut.model_validate(user)
