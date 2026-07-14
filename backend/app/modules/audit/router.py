import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.core.deps import DbSession
from app.modules.audit.schemas import AuditLogOut
from app.modules.audit.service import AuditService
from app.modules.rbac.catalog import Perm
from app.modules.rbac.deps import require_permission
from app.modules.users.models import User

router = APIRouter(prefix="/workspaces/{workspace_id}/audit", tags=["audit"])


@router.get("")
async def list_audit_logs(
    workspace_id: uuid.UUID,
    db: DbSession,
    user: Annotated[User, Depends(require_permission(Perm.AUDIT_READ))],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[AuditLogOut]:
    logs = await AuditService(db).list_for_workspace(workspace_id, limit)
    return [AuditLogOut.model_validate(log) for log in logs]
