import uuid

from httpx import AsyncClient

from tests.helpers import WORKSPACES, add_member, create_workspace, signup


async def _mkdir(
    client: AsyncClient,
    headers: dict[str, str],
    workspace_id: str,
    name: str,
    parent_id: str | None = None,
) -> dict[str, str]:
    resp = await client.post(
        f"{WORKSPACES}/{workspace_id}/folders",
        json={"name": name, "parent_id": parent_id},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    body: dict[str, str] = resp.json()
    return body


async def test_create_and_list_children(db_client: AsyncClient) -> None:
    owner = await signup(db_client, "owner@example.com")
    ws = await create_workspace(db_client, owner)

    root = await _mkdir(db_client, owner, ws, "Contracts")
    await _mkdir(db_client, owner, ws, "2026", parent_id=root["id"])

    top = await db_client.get(f"{WORKSPACES}/{ws}/folders", headers=owner)
    assert [f["name"] for f in top.json()] == ["Contracts"]

    children = await db_client.get(
        f"{WORKSPACES}/{ws}/folders", params={"parent_id": root["id"]}, headers=owner
    )
    assert [f["name"] for f in children.json()] == ["2026"]


async def test_duplicate_names_rejected_even_at_root(db_client: AsyncClient) -> None:
    owner = await signup(db_client, "owner@example.com")
    ws = await create_workspace(db_client, owner)
    await _mkdir(db_client, owner, ws, "Contracts")

    dup = await db_client.post(
        f"{WORKSPACES}/{ws}/folders", json={"name": "Contracts"}, headers=owner
    )
    assert dup.status_code == 409  # NULLS NOT DISTINCT makes root names unique too


async def test_same_name_allowed_under_different_parents(db_client: AsyncClient) -> None:
    owner = await signup(db_client, "owner@example.com")
    ws = await create_workspace(db_client, owner)
    a = await _mkdir(db_client, owner, ws, "A")
    b = await _mkdir(db_client, owner, ws, "B")

    await _mkdir(db_client, owner, ws, "Reports", parent_id=a["id"])
    await _mkdir(db_client, owner, ws, "Reports", parent_id=b["id"])  # no conflict


async def test_tree_returns_depth_and_path(db_client: AsyncClient) -> None:
    owner = await signup(db_client, "owner@example.com")
    ws = await create_workspace(db_client, owner)
    root = await _mkdir(db_client, owner, ws, "Contracts")
    child = await _mkdir(db_client, owner, ws, "2026", parent_id=root["id"])
    await _mkdir(db_client, owner, ws, "Q3", parent_id=child["id"])

    resp = await db_client.get(f"{WORKSPACES}/{ws}/folders/tree", headers=owner)

    assert resp.status_code == 200
    tree = resp.json()
    assert [(t["path"], t["depth"]) for t in tree] == [
        ("Contracts", 0),
        ("Contracts/2026", 1),
        ("Contracts/2026/Q3", 2),
    ]


async def test_move_and_cycle_prevention(db_client: AsyncClient) -> None:
    owner = await signup(db_client, "owner@example.com")
    ws = await create_workspace(db_client, owner)
    a = await _mkdir(db_client, owner, ws, "A")
    b = await _mkdir(db_client, owner, ws, "B", parent_id=a["id"])

    # moving A under its own child B must be rejected
    cycle = await db_client.patch(
        f"{WORKSPACES}/{ws}/folders/{a['id']}", json={"parent_id": b["id"]}, headers=owner
    )
    assert cycle.status_code == 409

    # moving B to the root is fine
    moved = await db_client.patch(
        f"{WORKSPACES}/{ws}/folders/{b['id']}", json={"parent_id": None}, headers=owner
    )
    assert moved.status_code == 200
    assert moved.json()["parent_id"] is None


async def test_delete_cascades_subtree(db_client: AsyncClient) -> None:
    owner = await signup(db_client, "owner@example.com")
    ws = await create_workspace(db_client, owner)
    root = await _mkdir(db_client, owner, ws, "Contracts")
    await _mkdir(db_client, owner, ws, "2026", parent_id=root["id"])

    resp = await db_client.delete(f"{WORKSPACES}/{ws}/folders/{root['id']}", headers=owner)
    assert resp.status_code == 204

    tree = await db_client.get(f"{WORKSPACES}/{ws}/folders/tree", headers=owner)
    assert tree.json() == []


async def test_viewer_can_read_but_not_create(db_client: AsyncClient) -> None:
    owner = await signup(db_client, "owner@example.com")
    viewer = await signup(db_client, "viewer@example.com")
    ws = await create_workspace(db_client, owner)
    await add_member(db_client, owner, ws, "viewer@example.com", "viewer")
    await _mkdir(db_client, owner, ws, "Contracts")

    read = await db_client.get(f"{WORKSPACES}/{ws}/folders/tree", headers=viewer)
    assert read.status_code == 200

    write = await db_client.post(
        f"{WORKSPACES}/{ws}/folders", json={"name": "Nope"}, headers=viewer
    )
    assert write.status_code == 403


async def test_parent_from_other_workspace_is_404(db_client: AsyncClient) -> None:
    owner = await signup(db_client, "owner@example.com")
    ws1 = await create_workspace(db_client, owner, name="One")
    ws2 = await create_workspace(db_client, owner, name="Two")
    foreign = await _mkdir(db_client, owner, ws2, "Elsewhere")

    resp = await db_client.post(
        f"{WORKSPACES}/{ws1}/folders",
        json={"name": "Child", "parent_id": foreign["id"]},
        headers=owner,
    )
    assert resp.status_code == 404


async def test_unknown_parent_uuid_is_404(db_client: AsyncClient) -> None:
    owner = await signup(db_client, "owner@example.com")
    ws = await create_workspace(db_client, owner)

    resp = await db_client.post(
        f"{WORKSPACES}/{ws}/folders",
        json={"name": "Child", "parent_id": str(uuid.uuid4())},
        headers=owner,
    )
    assert resp.status_code == 404
