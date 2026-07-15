from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.controller.auth_controller import auth_controller
from app.controller.auth_controller.dto.auth_dto import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
)
from app.controller.auth_controller.dto.user_dto import UserOut
from app.dependencies import CurrentUser, DbSession
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


def get_auth_service(db: DbSession) -> AuthService:
    return AuthService(db)


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(data: RegisterRequest, service: AuthServiceDep) -> UserOut:
    return await auth_controller.register(data, service)


@router.post("/login")
async def login(data: LoginRequest, service: AuthServiceDep) -> TokenResponse:
    return await auth_controller.login(data, service)


@router.post("/refresh")
async def refresh(data: RefreshRequest, service: AuthServiceDep) -> TokenResponse:
    return await auth_controller.refresh(data, service)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(data: RefreshRequest, service: AuthServiceDep) -> None:
    await auth_controller.logout(data, service)


@router.get("/me")
async def me(user: CurrentUser) -> UserOut:
    return auth_controller.me(user)
