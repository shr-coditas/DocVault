import uuid

from app.controller.audit_controller.dto.audit_dto import AuditLogOut
from app.services.audit_service import AuditService


async def list_audit_logs(
    workspace_id: uuid.UUID, limit: int, service: AuditService
) -> list[AuditLogOut]:
    logs = await service.list_for_workspace(workspace_id, limit)
    return [AuditLogOut.model_validate(log) for log in logs]
