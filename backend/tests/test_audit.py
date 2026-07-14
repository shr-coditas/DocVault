from httpx import AsyncClient

from tests.helpers import WORKSPACES, add_member, create_workspace, signup


async def test_mutations_are_audited(db_client: AsyncClient) -> None:
    owner = await signup(db_client, "owner@example.com")
    await signup(db_client, "member@example.com")
    workspace_id = await create_workspace(db_client, owner)
    await add_member(db_client, owner, workspace_id, "member@example.com", "editor")
    await db_client.post(
        f"{WORKSPACES}/{workspace_id}/teams", json={"name": "Platform"}, headers=owner
    )

    resp = await db_client.get(f"{WORKSPACES}/{workspace_id}/audit", headers=owner)

    assert resp.status_code == 200
    logs = resp.json()
    actions = [log["action"] for log in logs]
    assert "workspace.created" in actions
    assert "member.added" in actions
    assert "team.created" in actions

    created = next(log for log in logs if log["action"] == "workspace.created")
    assert created["actor_id"] is not None
    assert created["request_id"]  # correlates with the API access logs
    assert created["resource_type"] == "workspace"

    member_added = next(log for log in logs if log["action"] == "member.added")
    assert member_added["extra"]["email"] == "member@example.com"
    assert member_added["extra"]["role"] == "editor"
