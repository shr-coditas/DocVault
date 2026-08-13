import hashlib
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.db.base import Base
from app.db.session import get_db
from app.dependencies import get_storage_service
from app.main import create_app
from app.services.storage_service import StorageService, document_key
from tests.helpers import (
    WORKSPACES,
    add_member,
    create_schema,
    create_workspace,
    signup,
    sync_rbac_catalog,
)

pytestmark = pytest.mark.integration


@pytest.fixture
async def docs_client(
    postgres_url: str, test_storage: StorageService
) -> AsyncIterator[AsyncClient]:
    """API client wired to throwaway Postgres + MinIO, fresh schema per test."""
    engine = create_async_engine(postgres_url)
    await create_schema(engine)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        await sync_rbac_catalog(session)

    async def override_get_db() -> AsyncIterator[AsyncSession]:
        async with factory() as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_storage_service] = lambda: test_storage
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


def _docs_url(workspace_id: str) -> str:
    return f"{WORKSPACES}/{workspace_id}/documents"


async def _upload(
    client: AsyncClient,
    headers: dict[str, str],
    workspace_id: str,
    content: bytes = b"hello docvault",
    filename: str = "notes.txt",
    folder_id: str | None = None,
) -> dict:
    data = {"folder_id": folder_id} if folder_id else {}
    resp = await client.post(
        f"{_docs_url(workspace_id)}/upload",
        files={"file": (filename, content, "text/plain")},
        data=data,
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    body: dict = resp.json()
    return body


async def test_upload_returns_metadata(docs_client: AsyncClient) -> None:
    owner = await signup(docs_client, "owner@example.com")
    ws = await create_workspace(docs_client, owner)

    content = b"hello docvault"
    doc = await _upload(docs_client, owner, ws, content=content)

    assert doc["file_name"] == "notes.txt"
    assert doc["size_bytes"] == len(content)
    assert doc["checksum_sha256"] == hashlib.sha256(content).hexdigest()
    assert doc["mime_type"] == "text/plain"
    assert doc["search_status"] == "waiting_for_index"


async def test_download_returns_same_bytes(docs_client: AsyncClient) -> None:
    owner = await signup(docs_client, "owner@example.com")
    ws = await create_workspace(docs_client, owner)
    payload = b"the quick brown fox" * 1000
    doc = await _upload(docs_client, owner, ws, content=payload, filename="fox.txt")

    resp = await docs_client.get(f"{_docs_url(ws)}/{doc['id']}/download", headers=owner)
    assert resp.status_code == 200
    assert resp.content == payload


async def test_list_shows_uploaded_document(docs_client: AsyncClient) -> None:
    owner = await signup(docs_client, "owner@example.com")
    ws = await create_workspace(docs_client, owner)
    await _upload(docs_client, owner, ws, filename="a.txt")

    resp = await docs_client.get(_docs_url(ws), headers=owner)
    assert resp.status_code == 200
    assert [d["file_name"] for d in resp.json()] == ["a.txt"]


async def test_oversized_upload_rejected(
    docs_client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = await signup(docs_client, "owner@example.com")
    ws = await create_workspace(docs_client, owner)
    monkeypatch.setattr(get_settings(), "max_upload_size_bytes", 10)

    resp = await docs_client.post(
        f"{_docs_url(ws)}/upload",
        files={"file": ("big.txt", b"way more than ten bytes", "text/plain")},
        headers=owner,
    )
    assert resp.status_code == 413, resp.text


async def test_unsupported_file_type_rejected(docs_client: AsyncClient) -> None:
    owner = await signup(docs_client, "owner@example.com")
    ws = await create_workspace(docs_client, owner)

    resp = await docs_client.post(
        f"{_docs_url(ws)}/upload",
        files={"file": ("payload.exe", b"MZ...", "application/octet-stream")},
        headers=owner,
    )
    assert resp.status_code == 415, resp.text

    # the extension decides, not the declared content type
    disguised = await docs_client.post(
        f"{_docs_url(ws)}/upload",
        files={"file": ("payload.exe", b"MZ...", "text/plain")},
        headers=owner,
    )
    assert disguised.status_code == 415, disguised.text


async def test_supported_file_types_accepted(docs_client: AsyncClient) -> None:
    owner = await signup(docs_client, "owner@example.com")
    ws = await create_workspace(docs_client, owner)

    for name in ("report.pdf", "notes.MD", "rows.csv", "memo.docx"):
        resp = await docs_client.post(
            f"{_docs_url(ws)}/upload",
            files={"file": (name, b"content", "application/octet-stream")},
            headers=owner,
        )
        assert resp.status_code == 201, f"{name}: {resp.text}"


async def test_viewer_cannot_upload_but_can_download(docs_client: AsyncClient) -> None:
    owner = await signup(docs_client, "owner@example.com")
    viewer = await signup(docs_client, "viewer@example.com")
    ws = await create_workspace(docs_client, owner)
    await add_member(docs_client, owner, ws, "viewer@example.com", "viewer")
    doc = await _upload(docs_client, owner, ws, content=b"shared", filename="s.txt")

    blocked = await docs_client.post(
        f"{_docs_url(ws)}/upload",
        files={"file": ("nope.txt", b"nope", "text/plain")},
        headers=viewer,
    )
    assert blocked.status_code == 403

    got = await docs_client.get(f"{_docs_url(ws)}/{doc['id']}/download", headers=viewer)
    assert got.status_code == 200
    assert got.content == b"shared"


async def _object_exists(storage: StorageService, key: str) -> bool:
    try:
        async for _ in storage.stream(key):
            break
        return True
    except Exception:
        return False


async def test_rename_and_move(docs_client: AsyncClient) -> None:
    owner = await signup(docs_client, "owner@example.com")
    ws = await create_workspace(docs_client, owner)
    folder = await docs_client.post(
        f"{WORKSPACES}/{ws}/folders", json={"name": "Reports"}, headers=owner
    )
    folder_id = folder.json()["id"]
    doc = await _upload(docs_client, owner, ws, filename="draft.txt")

    renamed = await docs_client.patch(
        f"{_docs_url(ws)}/{doc['id']}",
        json={"title": "Final Report", "folder_id": folder_id},
        headers=owner,
    )
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["title"] == "Final Report"
    assert renamed.json()["folder_id"] == folder_id

    # move back to root
    to_root = await docs_client.patch(
        f"{_docs_url(ws)}/{doc['id']}", json={"folder_id": None}, headers=owner
    )
    assert to_root.json()["folder_id"] is None


async def test_list_scope_all_spans_folders(docs_client: AsyncClient) -> None:
    """Folder-scoped listing is the default; scope=all is how you see the lot.

    Without scope, an omitted folder_id means the workspace *root*, so anything
    filed in a folder is legitimately absent.
    """
    owner = await signup(docs_client, "owner@example.com")
    ws = await create_workspace(docs_client, owner)
    folder = await docs_client.post(
        f"{WORKSPACES}/{ws}/folders", json={"name": "Reports"}, headers=owner
    )
    folder_id = folder.json()["id"]
    root_doc = await _upload(docs_client, owner, ws, filename="root.txt")
    filed_doc = await _upload(docs_client, owner, ws, filename="filed.txt", folder_id=folder_id)

    root_only = await docs_client.get(_docs_url(ws), headers=owner)
    assert [d["id"] for d in root_only.json()] == [root_doc["id"]]

    in_folder = await docs_client.get(_docs_url(ws), params={"folder_id": folder_id}, headers=owner)
    assert [d["id"] for d in in_folder.json()] == [filed_doc["id"]]

    everything = await docs_client.get(_docs_url(ws), params={"scope": "all"}, headers=owner)
    assert {d["id"] for d in everything.json()} == {root_doc["id"], filed_doc["id"]}


async def test_list_scope_all_still_filters_by_visibility(docs_client: AsyncClient) -> None:
    owner = await signup(docs_client, "owner@example.com")
    viewer = await signup(docs_client, "viewer@example.com")
    ws = await create_workspace(docs_client, owner)
    await add_member(docs_client, owner, ws, "viewer@example.com", "viewer")
    folder = await docs_client.post(
        f"{WORKSPACES}/{ws}/folders", json={"name": "Reports"}, headers=owner
    )
    doc = await _upload(docs_client, owner, ws, filename="filed.txt", folder_id=folder.json()["id"])
    await _set_visibility(docs_client, owner, ws, doc["id"], "restricted")

    everything = await docs_client.get(_docs_url(ws), params={"scope": "all"}, headers=viewer)
    assert everything.json() == []


async def test_move_to_foreign_folder_is_404(docs_client: AsyncClient) -> None:
    owner = await signup(docs_client, "owner@example.com")
    ws1 = await create_workspace(docs_client, owner, name="One")
    ws2 = await create_workspace(docs_client, owner, name="Two")
    foreign = await docs_client.post(
        f"{WORKSPACES}/{ws2}/folders", json={"name": "Elsewhere"}, headers=owner
    )
    doc = await _upload(docs_client, owner, ws1)

    resp = await docs_client.patch(
        f"{_docs_url(ws1)}/{doc['id']}",
        json={"folder_id": foreign.json()["id"]},
        headers=owner,
    )
    assert resp.status_code == 404


async def test_trash_hides_and_restore_brings_back(docs_client: AsyncClient) -> None:
    owner = await signup(docs_client, "owner@example.com")
    ws = await create_workspace(docs_client, owner)
    doc = await _upload(docs_client, owner, ws, content=b"keep me", filename="keep.txt")

    trashed = await docs_client.delete(f"{_docs_url(ws)}/{doc['id']}", headers=owner)
    assert trashed.status_code == 204

    # hidden from list / get / download
    assert (await docs_client.get(_docs_url(ws), headers=owner)).json() == []
    assert (await docs_client.get(f"{_docs_url(ws)}/{doc['id']}", headers=owner)).status_code == 404
    assert (
        await docs_client.get(f"{_docs_url(ws)}/{doc['id']}/download", headers=owner)
    ).status_code == 404

    # visible in trash
    trash = await docs_client.get(f"{_docs_url(ws)}/trash", headers=owner)
    assert [d["id"] for d in trash.json()] == [doc["id"]]

    # restore → back to normal
    restored = await docs_client.post(f"{_docs_url(ws)}/{doc['id']}/restore", headers=owner)
    assert restored.status_code == 200
    assert restored.json()["deleted_at"] is None
    assert [d["id"] for d in (await docs_client.get(_docs_url(ws), headers=owner)).json()] == [
        doc["id"]
    ]
    download = await docs_client.get(f"{_docs_url(ws)}/{doc['id']}/download", headers=owner)
    assert download.content == b"keep me"


async def test_restore_active_document_is_404(docs_client: AsyncClient) -> None:
    owner = await signup(docs_client, "owner@example.com")
    ws = await create_workspace(docs_client, owner)
    doc = await _upload(docs_client, owner, ws)

    resp = await docs_client.post(f"{_docs_url(ws)}/{doc['id']}/restore", headers=owner)
    assert resp.status_code == 404


async def test_permanent_delete_removes_row_and_object(
    docs_client: AsyncClient, test_storage: StorageService
) -> None:
    owner = await signup(docs_client, "owner@example.com")
    ws = await create_workspace(docs_client, owner)
    doc = await _upload(docs_client, owner, ws, filename="gone.txt")
    key = document_key(ws, doc["id"], 1, "gone.txt")
    assert await _object_exists(test_storage, key)

    resp = await docs_client.delete(f"{_docs_url(ws)}/{doc['id']}/permanent", headers=owner)
    assert resp.status_code == 204

    trash = await docs_client.get(f"{_docs_url(ws)}/trash", headers=owner)
    assert trash.json() == []
    assert not await _object_exists(test_storage, key)


async def test_folder_delete_cleans_up_objects(
    docs_client: AsyncClient, test_storage: StorageService
) -> None:
    owner = await signup(docs_client, "owner@example.com")
    ws = await create_workspace(docs_client, owner)
    parent = await docs_client.post(f"{WORKSPACES}/{ws}/folders", json={"name": "P"}, headers=owner)
    child = await docs_client.post(
        f"{WORKSPACES}/{ws}/folders",
        json={"name": "C", "parent_id": parent.json()["id"]},
        headers=owner,
    )
    doc = await _upload(docs_client, owner, ws, filename="nested.txt", folder_id=child.json()["id"])
    key = document_key(ws, doc["id"], 1, "nested.txt")
    assert await _object_exists(test_storage, key)

    resp = await docs_client.delete(
        f"{WORKSPACES}/{ws}/folders/{parent.json()['id']}", headers=owner
    )
    assert resp.status_code == 204
    assert not await _object_exists(test_storage, key)  # no orphan left behind


async def test_workspace_delete_cleans_up_objects(
    docs_client: AsyncClient, test_storage: StorageService
) -> None:
    owner = await signup(docs_client, "owner@example.com")
    ws = await create_workspace(docs_client, owner)
    folder = await docs_client.post(
        f"{WORKSPACES}/{ws}/folders", json={"name": "Reports"}, headers=owner
    )
    root_doc = await _upload(docs_client, owner, ws, filename="root.txt")
    nested = await _upload(
        docs_client, owner, ws, filename="nested.txt", folder_id=folder.json()["id"]
    )
    trashed = await _upload(docs_client, owner, ws, filename="trashed.txt")
    await docs_client.delete(f"{_docs_url(ws)}/{trashed['id']}", headers=owner)  # to trash

    keys = [
        document_key(ws, doc["id"], 1, name)
        for doc, name in (
            (root_doc, "root.txt"),
            (nested, "nested.txt"),
            (trashed, "trashed.txt"),
        )
    ]
    for key in keys:
        assert await _object_exists(test_storage, key)

    resp = await docs_client.delete(f"{WORKSPACES}/{ws}", headers=owner)
    assert resp.status_code == 204

    # every object goes with the workspace - active, nested, and trashed alike
    for key in keys:
        assert not await _object_exists(test_storage, key)


async def test_viewer_cannot_trash_restore_or_delete(docs_client: AsyncClient) -> None:
    owner = await signup(docs_client, "owner@example.com")
    viewer = await signup(docs_client, "viewer@example.com")
    ws = await create_workspace(docs_client, owner)
    await add_member(docs_client, owner, ws, "viewer@example.com", "viewer")
    doc = await _upload(docs_client, owner, ws)

    assert (
        await docs_client.delete(f"{_docs_url(ws)}/{doc['id']}", headers=viewer)
    ).status_code == 403
    assert (
        await docs_client.post(f"{_docs_url(ws)}/{doc['id']}/restore", headers=viewer)
    ).status_code == 403
    assert (
        await docs_client.delete(f"{_docs_url(ws)}/{doc['id']}/permanent", headers=viewer)
    ).status_code == 403
    assert (await docs_client.get(f"{_docs_url(ws)}/trash", headers=viewer)).status_code == 403


async def test_upload_to_foreign_folder_is_404(docs_client: AsyncClient) -> None:
    owner = await signup(docs_client, "owner@example.com")
    ws1 = await create_workspace(docs_client, owner, name="One")
    ws2 = await create_workspace(docs_client, owner, name="Two")
    folder = await docs_client.post(
        f"{WORKSPACES}/{ws2}/folders", json={"name": "Elsewhere"}, headers=owner
    )
    foreign_folder_id = folder.json()["id"]

    resp = await docs_client.post(
        f"{_docs_url(ws1)}/upload",
        files={"file": ("x.txt", b"x", "text/plain")},
        data={"folder_id": foreign_folder_id},
        headers=owner,
    )
    assert resp.status_code == 404, resp.text


# -- visibility & sharing --------------------------------------------------

ME = "/api/v1/auth/me"


async def _user_id(client: AsyncClient, headers: dict[str, str]) -> str:
    resp = await client.get(ME, headers=headers)
    assert resp.status_code == 200, resp.text
    user_id: str = resp.json()["id"]
    return user_id


async def _set_visibility(
    client: AsyncClient, headers: dict[str, str], ws: str, doc_id: str, visibility: str
) -> None:
    resp = await client.put(
        f"{_docs_url(ws)}/{doc_id}/visibility", json={"visibility": visibility}, headers=headers
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["visibility"] == visibility


async def _can_see(client: AsyncClient, headers: dict[str, str], ws: str, doc_id: str) -> bool:
    resp = await client.get(f"{_docs_url(ws)}/{doc_id}", headers=headers)
    return resp.status_code == 200


async def test_default_visibility_is_workspace(docs_client: AsyncClient) -> None:
    owner = await signup(docs_client, "owner@example.com")
    ws = await create_workspace(docs_client, owner)
    doc = await _upload(docs_client, owner, ws)
    assert doc["visibility"] == "workspace"


async def test_restricted_document_hidden_from_other_member(docs_client: AsyncClient) -> None:
    owner = await signup(docs_client, "owner@example.com")
    viewer = await signup(docs_client, "viewer@example.com")
    ws = await create_workspace(docs_client, owner)
    await add_member(docs_client, owner, ws, "viewer@example.com", "viewer")
    doc = await _upload(docs_client, owner, ws, content=b"secret", filename="secret.txt")

    await _set_visibility(docs_client, owner, ws, doc["id"], "restricted")

    # invisible to the viewer: get, download, and list all hide it
    assert (
        await docs_client.get(f"{_docs_url(ws)}/{doc['id']}", headers=viewer)
    ).status_code == 404
    assert (
        await docs_client.get(f"{_docs_url(ws)}/{doc['id']}/download", headers=viewer)
    ).status_code == 404
    assert (await docs_client.get(_docs_url(ws), headers=viewer)).json() == []

    # the document owner still sees it
    assert await _can_see(docs_client, owner, ws, doc["id"])
    assert [d["id"] for d in (await docs_client.get(_docs_url(ws), headers=owner)).json()] == [
        doc["id"]
    ]


async def test_user_grant_reveals_then_removal_hides(docs_client: AsyncClient) -> None:
    owner = await signup(docs_client, "owner@example.com")
    viewer = await signup(docs_client, "viewer@example.com")
    ws = await create_workspace(docs_client, owner)
    await add_member(docs_client, owner, ws, "viewer@example.com", "viewer")
    viewer_id = await _user_id(docs_client, viewer)
    doc = await _upload(docs_client, owner, ws, content=b"secret", filename="s.txt")
    await _set_visibility(docs_client, owner, ws, doc["id"], "restricted")

    grant = await docs_client.post(
        f"{_docs_url(ws)}/{doc['id']}/grants",
        json={"principal_type": "user", "principal_id": viewer_id},
        headers=owner,
    )
    assert grant.status_code == 201, grant.text

    # now visible + downloadable
    assert await _can_see(docs_client, viewer, ws, doc["id"])
    got = await docs_client.get(f"{_docs_url(ws)}/{doc['id']}/download", headers=viewer)
    assert got.status_code == 200 and got.content == b"secret"

    # remove the grant → hidden again
    removed = await docs_client.delete(
        f"{_docs_url(ws)}/{doc['id']}/grants/user/{viewer_id}", headers=owner
    )
    assert removed.status_code == 204
    assert not await _can_see(docs_client, viewer, ws, doc["id"])


async def test_team_visibility_grants_whole_team(docs_client: AsyncClient) -> None:
    owner = await signup(docs_client, "owner@example.com")
    member = await signup(docs_client, "member@example.com")
    outsider = await signup(docs_client, "outsider@example.com")
    ws = await create_workspace(docs_client, owner)
    await add_member(docs_client, owner, ws, "member@example.com", "viewer")
    await add_member(docs_client, owner, ws, "outsider@example.com", "viewer")
    member_id = await _user_id(docs_client, member)

    team = await docs_client.post(
        f"{WORKSPACES}/{ws}/teams", json={"name": "Finance"}, headers=owner
    )
    team_id = team.json()["id"]
    added = await docs_client.post(
        f"{WORKSPACES}/{ws}/teams/{team_id}/members", json={"user_id": member_id}, headers=owner
    )
    assert added.status_code == 204, added.text

    doc = await _upload(docs_client, owner, ws, content=b"team-only", filename="t.txt")
    await _set_visibility(docs_client, owner, ws, doc["id"], "restricted")
    grant = await docs_client.post(
        f"{_docs_url(ws)}/{doc['id']}/grants",
        json={"principal_type": "team", "principal_id": team_id},
        headers=owner,
    )
    assert grant.status_code == 201, grant.text

    # team member sees it; the workspace member not on the team does not
    assert await _can_see(docs_client, member, ws, doc["id"])
    assert not await _can_see(docs_client, outsider, ws, doc["id"])


async def _make_team(
    client: AsyncClient, headers: dict[str, str], ws: str, name: str = "Finance"
) -> str:
    resp = await client.post(f"{WORKSPACES}/{ws}/teams", json={"name": name}, headers=headers)
    assert resp.status_code == 201, resp.text
    team_id: str = resp.json()["id"]
    return team_id


async def _add_to_team(
    client: AsyncClient, headers: dict[str, str], ws: str, team_id: str, user_id: str
) -> None:
    resp = await client.post(
        f"{WORKSPACES}/{ws}/teams/{team_id}/members", json={"user_id": user_id}, headers=headers
    )
    assert resp.status_code == 204, resp.text


async def _grant(
    client: AsyncClient,
    headers: dict[str, str],
    ws: str,
    doc_id: str,
    principal_type: str,
    principal_id: str,
) -> None:
    resp = await client.post(
        f"{_docs_url(ws)}/{doc_id}/grants",
        json={"principal_type": principal_type, "principal_id": principal_id},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text


async def test_joining_a_granted_team_later_grants_access(docs_client: AsyncClient) -> None:
    """Grant first, join second - the order users actually hit in practice.

    Access is resolved from team membership at query time, so a member added
    after the fact must see the document without anyone re-sharing it.
    """
    owner = await signup(docs_client, "owner@example.com")
    latecomer = await signup(docs_client, "late@example.com")
    ws = await create_workspace(docs_client, owner)
    await add_member(docs_client, owner, ws, "late@example.com", "viewer")
    latecomer_id = await _user_id(docs_client, latecomer)

    team_id = await _make_team(docs_client, owner, ws)
    doc = await _upload(docs_client, owner, ws, content=b"team-only", filename="t.txt")
    await _set_visibility(docs_client, owner, ws, doc["id"], "restricted")
    await _grant(docs_client, owner, ws, doc["id"], "team", team_id)

    assert not await _can_see(docs_client, latecomer, ws, doc["id"])

    await _add_to_team(docs_client, owner, ws, team_id, latecomer_id)

    assert await _can_see(docs_client, latecomer, ws, doc["id"])
    assert [d["id"] for d in (await docs_client.get(_docs_url(ws), headers=latecomer)).json()] == [
        doc["id"]
    ]
    got = await docs_client.get(f"{_docs_url(ws)}/{doc['id']}/download", headers=latecomer)
    assert got.status_code == 200 and got.content == b"team-only"

    # leaving the team takes the access away again
    removed = await docs_client.delete(
        f"{WORKSPACES}/{ws}/teams/{team_id}/members/{latecomer_id}", headers=owner
    )
    assert removed.status_code == 204
    assert not await _can_see(docs_client, latecomer, ws, doc["id"])


async def test_team_grant_works_on_a_restricted_document(docs_client: AsyncClient) -> None:
    """A grant is a grant: naming a team must not require `team` visibility."""
    owner = await signup(docs_client, "owner@example.com")
    member = await signup(docs_client, "member@example.com")
    ws = await create_workspace(docs_client, owner)
    await add_member(docs_client, owner, ws, "member@example.com", "viewer")
    member_id = await _user_id(docs_client, member)

    team_id = await _make_team(docs_client, owner, ws)
    await _add_to_team(docs_client, owner, ws, team_id, member_id)

    doc = await _upload(docs_client, owner, ws, content=b"secret", filename="s.txt")
    await _set_visibility(docs_client, owner, ws, doc["id"], "restricted")
    await _grant(docs_client, owner, ws, doc["id"], "team", team_id)

    assert await _can_see(docs_client, member, ws, doc["id"])


async def test_user_grant_works_on_a_team_document(docs_client: AsyncClient) -> None:
    """One extra person on one team document, without joining them to the team."""
    owner = await signup(docs_client, "owner@example.com")
    teammate = await signup(docs_client, "teammate@example.com")
    guest = await signup(docs_client, "guest@example.com")
    ws = await create_workspace(docs_client, owner)
    await add_member(docs_client, owner, ws, "teammate@example.com", "viewer")
    await add_member(docs_client, owner, ws, "guest@example.com", "viewer")
    teammate_id = await _user_id(docs_client, teammate)
    guest_id = await _user_id(docs_client, guest)

    team_id = await _make_team(docs_client, owner, ws)
    await _add_to_team(docs_client, owner, ws, team_id, teammate_id)

    doc = await _upload(docs_client, owner, ws, content=b"team-only", filename="t.txt")
    await _set_visibility(docs_client, owner, ws, doc["id"], "restricted")
    await _grant(docs_client, owner, ws, doc["id"], "team", team_id)
    await _grant(docs_client, owner, ws, doc["id"], "user", guest_id)

    # the team keeps its access and the individual gets theirs
    assert await _can_see(docs_client, teammate, ws, doc["id"])
    assert await _can_see(docs_client, guest, ws, doc["id"])


async def test_grants_do_not_survive_removing_the_member(docs_client: AsyncClient) -> None:
    """Grants name users by bare uuid, so nothing cascades - re-adding someone
    must not silently restore access they were granted before."""
    owner = await signup(docs_client, "owner@example.com")
    member = await signup(docs_client, "member@example.com")
    ws = await create_workspace(docs_client, owner)
    await add_member(docs_client, owner, ws, "member@example.com", "viewer")
    member_id = await _user_id(docs_client, member)

    doc = await _upload(docs_client, owner, ws, content=b"secret", filename="s.txt")
    await _set_visibility(docs_client, owner, ws, doc["id"], "restricted")
    await _grant(docs_client, owner, ws, doc["id"], "user", member_id)
    assert await _can_see(docs_client, member, ws, doc["id"])

    removed = await docs_client.delete(f"{WORKSPACES}/{ws}/members/{member_id}", headers=owner)
    assert removed.status_code == 204

    await add_member(docs_client, owner, ws, "member@example.com", "viewer")
    assert not await _can_see(docs_client, member, ws, doc["id"])
    assert (
        await docs_client.get(f"{_docs_url(ws)}/{doc['id']}/grants", headers=owner)
    ).json() == []


async def test_deleting_a_team_drops_its_grants(docs_client: AsyncClient) -> None:
    owner = await signup(docs_client, "owner@example.com")
    member = await signup(docs_client, "member@example.com")
    ws = await create_workspace(docs_client, owner)
    await add_member(docs_client, owner, ws, "member@example.com", "viewer")
    member_id = await _user_id(docs_client, member)

    team_id = await _make_team(docs_client, owner, ws)
    await _add_to_team(docs_client, owner, ws, team_id, member_id)
    doc = await _upload(docs_client, owner, ws, content=b"team-only", filename="t.txt")
    await _set_visibility(docs_client, owner, ws, doc["id"], "restricted")
    await _grant(docs_client, owner, ws, doc["id"], "team", team_id)
    assert await _can_see(docs_client, member, ws, doc["id"])

    deleted = await docs_client.delete(f"{WORKSPACES}/{ws}/teams/{team_id}", headers=owner)
    assert deleted.status_code == 204

    assert not await _can_see(docs_client, member, ws, doc["id"])
    assert (
        await docs_client.get(f"{_docs_url(ws)}/{doc['id']}/grants", headers=owner)
    ).json() == []


async def test_workspace_owner_overrides_visibility(docs_client: AsyncClient) -> None:
    owner = await signup(docs_client, "owner@example.com")
    coowner = await signup(docs_client, "coowner@example.com")
    ws = await create_workspace(docs_client, owner)
    await add_member(docs_client, owner, ws, "coowner@example.com", "owner")
    doc = await _upload(docs_client, owner, ws, content=b"secret", filename="s.txt")
    await _set_visibility(docs_client, owner, ws, doc["id"], "restricted")

    # a second workspace owner sees the private document without an explicit grant
    assert await _can_see(docs_client, coowner, ws, doc["id"])
    assert [d["id"] for d in (await docs_client.get(_docs_url(ws), headers=coowner)).json()] == [
        doc["id"]
    ]


async def test_share_endpoints_require_share_permission(docs_client: AsyncClient) -> None:
    owner = await signup(docs_client, "owner@example.com")
    viewer = await signup(docs_client, "viewer@example.com")
    ws = await create_workspace(docs_client, owner)
    await add_member(docs_client, owner, ws, "viewer@example.com", "viewer")
    viewer_id = await _user_id(docs_client, viewer)
    doc = await _upload(docs_client, owner, ws)

    # viewer lacks document:share
    assert (
        await docs_client.put(
            f"{_docs_url(ws)}/{doc['id']}/visibility",
            json={"visibility": "restricted"},
            headers=viewer,
        )
    ).status_code == 403
    assert (
        await docs_client.get(f"{_docs_url(ws)}/{doc['id']}/grants", headers=viewer)
    ).status_code == 403
    assert (
        await docs_client.post(
            f"{_docs_url(ws)}/{doc['id']}/grants",
            json={"principal_type": "user", "principal_id": viewer_id},
            headers=viewer,
        )
    ).status_code == 403


async def test_grant_for_non_member_or_foreign_team_is_404(docs_client: AsyncClient) -> None:
    owner = await signup(docs_client, "owner@example.com")
    stranger = await signup(docs_client, "stranger@example.com")  # never added to ws
    ws = await create_workspace(docs_client, owner)
    other_ws = await create_workspace(docs_client, owner, name="Other")
    stranger_id = await _user_id(docs_client, stranger)
    foreign_team = await docs_client.post(
        f"{WORKSPACES}/{other_ws}/teams", json={"name": "Elsewhere"}, headers=owner
    )
    doc = await _upload(docs_client, owner, ws)

    non_member = await docs_client.post(
        f"{_docs_url(ws)}/{doc['id']}/grants",
        json={"principal_type": "user", "principal_id": stranger_id},
        headers=owner,
    )
    assert non_member.status_code == 404

    foreign = await docs_client.post(
        f"{_docs_url(ws)}/{doc['id']}/grants",
        json={"principal_type": "team", "principal_id": foreign_team.json()["id"]},
        headers=owner,
    )
    assert foreign.status_code == 404
