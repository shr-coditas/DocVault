import uuid
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.audit.models import AuditLog


class AuditService:
    """Records audit rows inside the caller's transaction.

    `record()` only stages the row; it is committed (or rolled back) together
    with the mutation it describes, so the trail can never disagree with the
    data.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def record(
        self,
        *,
        action: str,
        resource_type: str,
        resource_id: uuid.UUID | None = None,
        workspace_id: uuid.UUID | None = None,
        actor_id: uuid.UUID | None = None,
        **extra: Any,
    ) -> None:
        request_id = structlog.contextvars.get_contextvars().get("request_id")
        self.session.add(
            AuditLog(
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                workspace_id=workspace_id,
                actor_id=actor_id,
                request_id=request_id,
                extra=extra,
            )
        )

    async def list_for_workspace(self, workspace_id: uuid.UUID, limit: int = 50) -> list[AuditLog]:
        stmt = (
            select(AuditLog)
            .where(AuditLog.workspace_id == workspace_id)
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        return list((await self.session.execute(stmt)).scalars())
