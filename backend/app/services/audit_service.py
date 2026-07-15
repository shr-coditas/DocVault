import uuid
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit_log import AuditLog
from app.repository.audit_repository import AuditRepository


class AuditService:
    """Records audit rows inside the caller's transaction.

    `record()` only stages the row; it is committed (or rolled back) together
    with the mutation it describes, so the trail can never disagree with the
    data.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = AuditRepository(session)

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
        self.repository.add(
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
        return await self.repository.list_for_workspace(workspace_id, limit)
