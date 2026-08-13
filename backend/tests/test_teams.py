from httpx import AsyncClient

from tests.helpers import WORKSPACES, add_member, create_workspace, signup


async def _user_id(client: AsyncClient, headers: dict[str, str]) -> str:
    me = await client.get("/api/v1/auth/me", headers=headers)
    user_id: str = me.json()["id"]
    return user_id


async def test_create_and_list_teams(db_client: AsyncClient) -> None:
    owner = await signup(db_client, "owner@example.com")
    workspace_id = await create_workspace(db_client, owner)

    created = await db_client.post(
        f"{WORKSPACES}/{workspace_id}/teams", json={"name": "Platform"}, headers=owner
    )
    assert created.status_code == 201

    listed = await db_client.get(f"{WORKSPACES}/{workspace_id}/teams", headers=owner)
    assert [t["name"] for t in listed.json()] == ["Platform"]


async def test_creator_is_a_member_of_the_team_they_create(db_client: AsyncClient) -> None:
    """You are in the team you create - an empty team reads as a broken one."""
    owner = await signup(db_client, "owner@example.com")
    workspace_id = await create_workspace(db_client, owner)
    team_id = (
        await db_client.post(
            f"{WORKSPACES}/{workspace_id}/teams", json={"name": "Platform"}, headers=owner
        )
    ).json()["id"]

    listed = await db_client.get(
        f"{WORKSPACES}/{workspace_id}/teams/{team_id}/members", headers=owner
    )
    assert [m["email"] for m in listed.json()] == ["owner@example.com"]

    # and they can step back out, for the admin who set the team up for others
    owner_id = await _user_id(db_client, owner)
    removed = await db_client.delete(
        f"{WORKSPACES}/{workspace_id}/teams/{team_id}/members/{owner_id}", headers=owner
    )
    assert removed.status_code == 204
    emptied = await db_client.get(
        f"{WORKSPACES}/{workspace_id}/teams/{team_id}/members", headers=owner
    )
    assert emptied.json() == []


async def test_duplicate_team_name_in_workspace_is_409(db_client: AsyncClient) -> None:
    owner = await signup(db_client, "owner@example.com")
    workspace_id = await create_workspace(db_client, owner)
    await db_client.post(
        f"{WORKSPACES}/{workspace_id}/teams", json={"name": "Platform"}, headers=owner
    )

    dup = await db_client.post(
        f"{WORKSPACES}/{workspace_id}/teams", json={"name": "Platform"}, headers=owner
    )
    assert dup.status_code == 409


async def test_team_member_must_be_workspace_member(db_client: AsyncClient) -> None:
    owner = await signup(db_client, "owner@example.com")
    outsider = await signup(db_client, "outsider@example.com")
    workspace_id = await create_workspace(db_client, owner)
    team = await db_client.post(
        f"{WORKSPACES}/{workspace_id}/teams", json={"name": "Platform"}, headers=owner
    )
    team_id = team.json()["id"]
    outsider_id = await _user_id(db_client, outsider)

    resp = await db_client.post(
        f"{WORKSPACES}/{workspace_id}/teams/{team_id}/members",
        json={"user_id": outsider_id},
        headers=owner,
    )
    assert resp.status_code == 409


async def test_add_list_and_remove_team_member(db_client: AsyncClient) -> None:
    owner = await signup(db_client, "owner@example.com")
    member = await signup(db_client, "member@example.com")
    workspace_id = await create_workspace(db_client, owner)
    await add_member(db_client, owner, workspace_id, "member@example.com", "editor")

    team_id = (
        await db_client.post(
            f"{WORKSPACES}/{workspace_id}/teams", json={"name": "Platform"}, headers=owner
        )
    ).json()["id"]
    member_id = await _user_id(db_client, member)

    added = await db_client.post(
        f"{WORKSPACES}/{workspace_id}/teams/{team_id}/members",
        json={"user_id": member_id},
        headers=owner,
    )
    assert added.status_code == 204

    listed = await db_client.get(
        f"{WORKSPACES}/{workspace_id}/teams/{team_id}/members", headers=owner
    )
    # the creator is seeded in first, then the member we just added
    assert [m["email"] for m in listed.json()] == ["owner@example.com", "member@example.com"]

    removed = await db_client.delete(
        f"{WORKSPACES}/{workspace_id}/teams/{team_id}/members/{member_id}", headers=owner
    )
    assert removed.status_code == 204


async def test_team_under_wrong_workspace_is_404(db_client: AsyncClient) -> None:
    owner = await signup(db_client, "owner@example.com")
    ws1 = await create_workspace(db_client, owner, name="One")
    ws2 = await create_workspace(db_client, owner, name="Two")
    team_id = (
        await db_client.post(f"{WORKSPACES}/{ws1}/teams", json={"name": "Platform"}, headers=owner)
    ).json()["id"]

    resp = await db_client.delete(f"{WORKSPACES}/{ws2}/teams/{team_id}", headers=owner)
    assert resp.status_code == 404
