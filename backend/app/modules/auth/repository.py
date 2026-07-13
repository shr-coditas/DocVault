import uuid

from sqlalchemy import func, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.auth.models import RefreshToken


class RefreshTokenRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, jti: uuid.UUID) -> RefreshToken | None:
        return await self.session.get(RefreshToken, jti)

    def add(self, token: RefreshToken) -> None:
        self.session.add(token)

    async def revoke_family(self, family_id: uuid.UUID) -> None:
        await self.session.execute(
            update(RefreshToken)
            .where(RefreshToken.family_id == family_id, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=func.now())
        )
