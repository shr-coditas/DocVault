"""All SQLAlchemy models.

Importing this package registers every table on Base.metadata — Alembic's
env.py and the test conftest rely on `from app import models` being enough.
"""

from app.models.audit_log import AuditLog
from app.models.document import Document
from app.models.document_grant import DocumentGrant
from app.models.folder import Folder
from app.models.rbac import Permission, Role, role_permissions
from app.models.refresh_token import RefreshToken
from app.models.team import Team, TeamMember
from app.models.user import User
from app.models.workspace import Workspace, WorkspaceMember

__all__ = [
    "AuditLog",
    "Document",
    "DocumentGrant",
    "Folder",
    "Permission",
    "RefreshToken",
    "Role",
    "Team",
    "TeamMember",
    "User",
    "Workspace",
    "WorkspaceMember",
    "role_permissions",
]
