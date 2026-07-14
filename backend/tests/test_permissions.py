"""The permission matrix, exercised end-to-end per role."""

from collections.abc import Awaitable, Callable

import pytest
from httpx import AsyncClient, Response

from tests.helpers import WORKSPACES, add_member, create_workspace, signup

Action = Callable[[AsyncClient, dict[str, str], str], Awaitable[Response]]


async def _get_workspace(c: AsyncClient, h: dict[str, str], ws: str) -> Response:
    return await c.get(f"{WORKSPACES}/{ws}", headers=h)


async def _update_workspace(c: AsyncClient, h: dict[str, str], ws: str) -> Response:
    return await c.patch(f"{WORKSPACES}/{ws}", json={"name": "Renamed"}, headers=h)


async def _delete_workspace(c: AsyncClient, h: dict[str, str], ws: str) -> Response:
    return await c.delete(f"{WORKSPACES}/{ws}", headers=h)


async def _add_member(c: AsyncClient, h: dict[str, str], ws: str) -> Response:
    await signup(c, "newbie@example.com")
    return await add_member(c, h, ws, "newbie@example.com", "viewer")


async def _create_team(c: AsyncClient, h: dict[str, str], ws: str) -> Response:
    return await c.post(f"{WORKSPACES}/{ws}/teams", json={"name": "Platform"}, headers=h)


async def _list_teams(c: AsyncClient, h: dict[str, str], ws: str) -> Response:
    return await c.get(f"{WORKSPACES}/{ws}/teams", headers=h)


async def _read_audit(c: AsyncClient, h: dict[str, str], ws: str) -> Response:
    return await c.get(f"{WORKSPACES}/{ws}/audit", headers=h)


ACTIONS: dict[str, Action] = {
    "read_workspace": _get_workspace,
    "update_workspace": _update_workspace,
    "delete_workspace": _delete_workspace,
    "add_member": _add_member,
    "create_team": _create_team,
    "list_teams": _list_teams,
    "read_audit": _read_audit,
}

MATRIX = [
    ("owner", "read_workspace", 200),
    ("editor", "read_workspace", 200),
    ("viewer", "read_workspace", 200),
    ("owner", "update_workspace", 200),
    ("editor", "update_workspace", 403),
    ("viewer", "update_workspace", 403),
    ("owner", "delete_workspace", 204),
    ("editor", "delete_workspace", 403),
    ("viewer", "delete_workspace", 403),
    ("owner", "add_member", 201),
    ("editor", "add_member", 403),
    ("viewer", "add_member", 403),
    ("owner", "create_team", 201),
    ("editor", "create_team", 403),
    ("viewer", "create_team", 403),
    ("owner", "list_teams", 200),
    ("editor", "list_teams", 200),
    ("viewer", "list_teams", 200),
    ("owner", "read_audit", 200),
    ("editor", "read_audit", 403),
    ("viewer", "read_audit", 403),
]


@pytest.mark.parametrize(("role", "action", "expected"), MATRIX)
async def test_permission_matrix(
    db_client: AsyncClient, role: str, action: str, expected: int
) -> None:
    owner_headers = await signup(db_client, "boss@example.com")
    workspace_id = await create_workspace(db_client, owner_headers)

    if role == "owner":
        actor_headers = owner_headers
    else:
        actor_headers = await signup(db_client, f"{role}@example.com")
        resp = await add_member(db_client, owner_headers, workspace_id, f"{role}@example.com", role)
        assert resp.status_code == 201

    resp = await ACTIONS[action](db_client, actor_headers, workspace_id)
    assert resp.status_code == expected, f"{role} {action}: {resp.text}"
