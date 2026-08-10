from httpx import AsyncClient, Response
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

# Importing the package is what registers every table on Base.metadata. Without
# it `create_schema` below silently creates nothing: create_all on empty metadata
# is a successful no-op, so the failure surfaces much later, and far away, as
# "relation does not exist".
from app import models  # noqa: F401
from app.db.base import Base
from app.models.rbac import Permission, Role
from app.utils.rbac_catalog import ROLE_PERMISSIONS

POSTGRES_IMAGE = "pgvector/pgvector:pg17"

REGISTER = "/api/v1/auth/register"
LOGIN = "/api/v1/auth/login"
WORKSPACES = "/api/v1/workspaces"

PASSWORD = "s3cret-password"


async def create_schema(engine: AsyncEngine) -> None:
    """Build the whole schema on a throwaway database.

    Every schema-building fixture must go through here. `document_chunks` has a
    `vector` column, so the extension has to exist before `create_all` runs - a
    fixture that calls `create_all` directly fails on that column, and the failure
    looks nothing like a missing extension.
    """
    assert Base.metadata.tables, "no tables registered - is app.models imported?"
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


async def signup(client: AsyncClient, email: str, full_name: str = "Test User") -> dict[str, str]:
    """Register + login a user; returns Authorization headers."""
    resp = await client.post(
        REGISTER, json={"email": email, "password": PASSWORD, "full_name": full_name}
    )
    assert resp.status_code == 201, resp.text
    resp = await client.post(LOGIN, json={"email": email, "password": PASSWORD})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


async def create_workspace(client: AsyncClient, headers: dict[str, str], name: str = "Acme") -> str:
    resp = await client.post(WORKSPACES, json={"name": name}, headers=headers)
    assert resp.status_code == 201, resp.text
    workspace_id: str = resp.json()["id"]
    return workspace_id


async def add_member(
    client: AsyncClient, headers: dict[str, str], workspace_id: str, email: str, role: str
) -> Response:
    return await client.post(
        f"{WORKSPACES}/{workspace_id}/members",
        json={"email": email, "role": role},
        headers=headers,
    )


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
