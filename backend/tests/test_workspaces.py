from httpx import AsyncClient

from tests.helpers import WORKSPACES, add_member, create_workspace, signup


async def test_create_workspace_makes_creator_owner(db_client: AsyncClient) -> None:
    headers = await signup(db_client, "owner@example.com")
    workspace_id = await create_workspace(db_client, headers)

    resp = await db_client.get(f"{WORKSPACES}/{workspace_id}/members", headers=headers)

    assert resp.status_code == 200
    members = resp.json()
    assert len(members) == 1
    assert members[0]["email"] == "owner@example.com"
    assert members[0]["role"] == "owner"


async def test_list_returns_only_my_workspaces(db_client: AsyncClient) -> None:
    mine = await signup(db_client, "mine@example.com")
    other = await signup(db_client, "other@example.com")
    my_ws = await create_workspace(db_client, mine, name="Mine")
    await create_workspace(db_client, other, name="Theirs")

    resp = await db_client.get(WORKSPACES, headers=mine)

    assert resp.status_code == 200
    body = resp.json()
    assert [ws["id"] for ws in body] == [my_ws]
    assert body[0]["my_role"] == "owner"


async def test_non_member_gets_404_not_403(db_client: AsyncClient) -> None:
    owner = await signup(db_client, "owner@example.com")
    outsider = await signup(db_client, "outsider@example.com")
    workspace_id = await create_workspace(db_client, owner)

    resp = await db_client.get(f"{WORKSPACES}/{workspace_id}", headers=outsider)

    assert resp.status_code == 404  # outsiders must not learn the workspace exists


async def test_owner_can_update_workspace(db_client: AsyncClient) -> None:
    headers = await signup(db_client, "owner@example.com")
    workspace_id = await create_workspace(db_client, headers)

    resp = await db_client.patch(
        f"{WORKSPACES}/{workspace_id}", json={"name": "Renamed"}, headers=headers
    )

    assert resp.status_code == 200
    assert resp.json()["name"] == "Renamed"


async def test_add_member_unknown_email_is_404(db_client: AsyncClient) -> None:
    headers = await signup(db_client, "owner@example.com")
    workspace_id = await create_workspace(db_client, headers)

    resp = await add_member(db_client, headers, workspace_id, "ghost@example.com", "viewer")
    assert resp.status_code == 404


async def test_add_member_twice_is_409(db_client: AsyncClient) -> None:
    owner = await signup(db_client, "owner@example.com")
    await signup(db_client, "member@example.com")
    workspace_id = await create_workspace(db_client, owner)

    first = await add_member(db_client, owner, workspace_id, "member@example.com", "viewer")
    assert first.status_code == 201
    second = await add_member(db_client, owner, workspace_id, "member@example.com", "viewer")
    assert second.status_code == 409


async def test_last_owner_cannot_be_demoted_or_removed(db_client: AsyncClient) -> None:
    owner = await signup(db_client, "owner@example.com")
    workspace_id = await create_workspace(db_client, owner)
    me = await db_client.get("/api/v1/auth/me", headers=owner)
    owner_id = me.json()["id"]

    demote = await db_client.patch(
        f"{WORKSPACES}/{workspace_id}/members/{owner_id}",
        json={"role": "viewer"},
        headers=owner,
    )
    assert demote.status_code == 409

    remove = await db_client.delete(
        f"{WORKSPACES}/{workspace_id}/members/{owner_id}", headers=owner
    )
    assert remove.status_code == 409


async def test_member_can_be_removed(db_client: AsyncClient) -> None:
    owner = await signup(db_client, "owner@example.com")
    await signup(db_client, "member@example.com")
    workspace_id = await create_workspace(db_client, owner)
    await add_member(db_client, owner, workspace_id, "member@example.com", "viewer")

    members = (await db_client.get(f"{WORKSPACES}/{workspace_id}/members", headers=owner)).json()
    target = next(m for m in members if m["email"] == "member@example.com")

    resp = await db_client.delete(
        f"{WORKSPACES}/{workspace_id}/members/{target['user_id']}", headers=owner
    )
    assert resp.status_code == 204

    remaining = (await db_client.get(f"{WORKSPACES}/{workspace_id}/members", headers=owner)).json()
    assert len(remaining) == 1
