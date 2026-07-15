from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.rbac import Permission, Role
from app.utils.rbac_catalog import ROLE_PERMISSIONS


async def sync_rbac_catalog(session: AsyncSession) -> None:
    """Idempotently make the roles/permissions tables match the catalog.

    Production databases are seeded by the migration; this is used by tests
    (which build their schema with create_all) and stays available as a repair
    tool.
    """
    existing_perms = {p.code: p for p in (await session.execute(select(Permission))).scalars()}
    all_codes = {code for perms in ROLE_PERMISSIONS.values() for code in perms}
    for code in sorted(all_codes):
        if code not in existing_perms:
            perm = Permission(code=code)
            session.add(perm)
            existing_perms[code] = perm

    existing_roles = {r.name: r for r in (await session.execute(select(Role))).scalars()}
    for role_name, codes in ROLE_PERMISSIONS.items():
        role = existing_roles.get(role_name)
        if role is None:
            role = Role(name=role_name)
            session.add(role)
        role.permissions = [existing_perms[code] for code in sorted(codes)]

    await session.commit()
