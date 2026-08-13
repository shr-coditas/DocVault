import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column
from uuid6 import uuid7

from app.db.base import Base


class RefreshToken(Base):
    """One row per issued refresh token.

    Tokens rotate on every use; rows in one `family_id` descend from a single
    login. A revoked token being presented again means theft/reuse - the whole
    family gets revoked.
    """

    __tablename__ = "refresh_tokens"

    jti: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid7)
    user_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    token_hash: Mapped[str] = mapped_column(sa.String(64))  # sha256 hex of the secret
    family_id: Mapped[uuid.UUID] = mapped_column(index=True)
    expires_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    replaced_by: Mapped[uuid.UUID | None]
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now()
    )
