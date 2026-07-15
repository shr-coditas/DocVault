import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from uuid6 import uuid7

from app.config import get_settings
from app.controller.auth_controller.dto.auth_dto import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
)
from app.exceptions import ConflictError, UnauthorizedError
from app.models.refresh_token import RefreshToken
from app.models.user import User
from app.repository.refresh_token_repository import RefreshTokenRepository
from app.repository.user_repository import UserRepository
from app.utils.security import (
    create_access_token,
    generate_refresh_secret,
    hash_password,
    hash_refresh_secret,
    verify_password,
)


def _invalid_credentials() -> UnauthorizedError:
    # same error for unknown email and wrong password: no account enumeration
    return UnauthorizedError("invalid credentials")


def _invalid_refresh() -> UnauthorizedError:
    return UnauthorizedError("invalid refresh token")


class AuthService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.users = UserRepository(session)
        self.refresh_tokens = RefreshTokenRepository(session)
        self.settings = get_settings()

    async def register(self, data: RegisterRequest) -> User:
        email = data.email.lower()
        if await self.users.get_by_email(email) is not None:
            raise ConflictError("email already registered")

        user = User(
            email=email,
            hashed_password=hash_password(data.password),
            full_name=data.full_name,
        )
        self.users.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def login(self, data: LoginRequest) -> TokenResponse:
        user = await self.users.get_by_email(data.email.lower())
        if (
            user is None
            or not verify_password(data.password, user.hashed_password)
            or not user.is_active
        ):
            raise _invalid_credentials()

        pair, _ = self._issue_pair(user.id, family_id=uuid.uuid4())
        await self.session.commit()
        return pair

    async def refresh(self, token_str: str) -> TokenResponse:
        row = await self._load_valid_row(token_str)

        user = await self.users.get(row.user_id)
        if user is None or not user.is_active:
            await self.refresh_tokens.revoke_family(row.family_id)
            await self.session.commit()
            raise _invalid_refresh()

        pair, new_jti = self._issue_pair(user.id, family_id=row.family_id)
        row.revoked_at = datetime.now(UTC)
        row.replaced_by = new_jti
        await self.session.commit()
        return pair

    async def logout(self, token_str: str) -> None:
        """Revoke the whole token family. Idempotent: bad tokens are ignored."""
        try:
            row = await self._load_valid_row(token_str)
        except UnauthorizedError:
            return
        await self.refresh_tokens.revoke_family(row.family_id)
        await self.session.commit()

    def _issue_pair(
        self, user_id: uuid.UUID, family_id: uuid.UUID
    ) -> tuple[TokenResponse, uuid.UUID]:
        access = create_access_token(
            user_id, self.settings.jwt_secret, self.settings.access_token_ttl_minutes
        )
        jti = uuid7()
        secret_value = generate_refresh_secret()
        self.refresh_tokens.add(
            RefreshToken(
                jti=jti,
                user_id=user_id,
                token_hash=hash_refresh_secret(secret_value),
                family_id=family_id,
                expires_at=datetime.now(UTC) + timedelta(days=self.settings.refresh_token_ttl_days),
            )
        )
        pair = TokenResponse(access_token=access, refresh_token=f"{jti}.{secret_value}")
        return pair, jti

    async def _load_valid_row(self, token_str: str) -> RefreshToken:
        jti_str, sep, secret_value = token_str.partition(".")
        if not sep:
            raise _invalid_refresh()
        try:
            jti = uuid.UUID(jti_str)
        except ValueError:
            raise _invalid_refresh() from None

        row = await self.refresh_tokens.get(jti)
        if row is None:
            raise _invalid_refresh()
        if row.revoked_at is not None:
            # a rotated-away token came back: assume theft, kill the family
            await self.refresh_tokens.revoke_family(row.family_id)
            await self.session.commit()
            raise _invalid_refresh()
        if row.token_hash != hash_refresh_secret(secret_value):
            raise _invalid_refresh()
        if row.expires_at <= datetime.now(UTC):
            raise _invalid_refresh()
        return row
