import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.controller.audit_controller import audit_controller
from app.controller.audit_controller.dto.audit_dto import AuditLogOut
from app.dependencies import DbSession, require_permission
from app.models.user import User
from app.services.audit_service import AuditService
from app.utils.rbac_catalog import Perm

router = APIRouter(prefix="/workspaces/{workspace_id}/audit", tags=["audit"])


def get_audit_service(db: DbSession) -> AuditService:
    return AuditService(db)


ServiceDep = Annotated[AuditService, Depends(get_audit_service)]


@router.get("")
async def list_audit_logs(
    workspace_id: uuid.UUID,
    user: Annotated[User, Depends(require_permission(Perm.AUDIT_READ))],
    service: ServiceDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[AuditLogOut]:
    return await audit_controller.list_audit_logs(workspace_id, limit, service)
