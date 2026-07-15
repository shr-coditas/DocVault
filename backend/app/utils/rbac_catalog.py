"""The RBAC catalog: permission codes and the role → permission matrix.

The database is the runtime source of truth (seeded from this catalog by the
migration); this module is the single place the catalog is defined in code.
"""

from enum import StrEnum


class Perm(StrEnum):
    WORKSPACE_READ = "workspace:read"
    WORKSPACE_UPDATE = "workspace:update"
    WORKSPACE_DELETE = "workspace:delete"
    WORKSPACE_MANAGE_MEMBERS = "workspace:manage_members"
    TEAM_READ = "team:read"
    TEAM_MANAGE = "team:manage"
    DOCUMENT_CREATE = "document:create"
    DOCUMENT_READ = "document:read"
    DOCUMENT_UPDATE = "document:update"
    DOCUMENT_DELETE = "document:delete"
    DOCUMENT_SHARE = "document:share"
    AUDIT_READ = "audit:read"


OWNER = "owner"
EDITOR = "editor"
VIEWER = "viewer"

ROLE_PERMISSIONS: dict[str, frozenset[Perm]] = {
    OWNER: frozenset(Perm),
    EDITOR: frozenset(
        {
            Perm.WORKSPACE_READ,
            Perm.TEAM_READ,
            Perm.DOCUMENT_CREATE,
            Perm.DOCUMENT_READ,
            Perm.DOCUMENT_UPDATE,
            Perm.DOCUMENT_DELETE,
            Perm.DOCUMENT_SHARE,
        }
    ),
    VIEWER: frozenset({Perm.WORKSPACE_READ, Perm.TEAM_READ, Perm.DOCUMENT_READ}),
}

ROLE_NAMES = tuple(ROLE_PERMISSIONS)
