from app.controller.auth_controller.dto.auth_dto import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
)
from app.controller.auth_controller.dto.user_dto import UserOut
from app.models.user import User
from app.services.auth_service import AuthService


async def register(data: RegisterRequest, service: AuthService) -> UserOut:
    user = await service.register(data)
    return UserOut.model_validate(user)


async def login(data: LoginRequest, service: AuthService) -> TokenResponse:
    return await service.login(data)


async def refresh(data: RefreshRequest, service: AuthService) -> TokenResponse:
    return await service.refresh(data.refresh_token)


async def logout(data: RefreshRequest, service: AuthService) -> None:
    await service.logout(data.refresh_token)


def me(user: User) -> UserOut:
    return UserOut.model_validate(user)
