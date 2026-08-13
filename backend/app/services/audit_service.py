import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit_log import AuditLog
from app.repository.audit_repository import AuditRepository
from app.services.activity_broadcaster import stage_activity_event


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
        if workspace_id is not None:
            # mirrored onto the live activity feed - published only post-commit
            stage_activity_event(
                self.session,
                {
                    "action": action,
                    "resource_type": resource_type,
                    "resource_id": str(resource_id) if resource_id else None,
                    "workspace_id": str(workspace_id),
                    "actor_id": str(actor_id) if actor_id else None,
                    "request_id": request_id,
                    "extra": extra,
                    "occurred_at": datetime.now(UTC).isoformat(),
                },
            )

    async def list_for_workspace(self, workspace_id: uuid.UUID, limit: int = 50) -> list[AuditLog]:
        return await self.repository.list_for_workspace(workspace_id, limit)
