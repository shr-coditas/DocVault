import hashlib
from collections.abc import AsyncIterator, Iterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.minio import MinioContainer

from app.config import Settings, get_settings
from app.db.base import Base
from app.db.session import get_db
from app.main import create_app
from app.routers.document_router import get_storage_service
from app.scripts.seed_rbac import sync_rbac_catalog
from app.services.storage_service import StorageService
from tests.helpers import WORKSPACES, add_member, create_workspace, signup

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def storage_settings() -> Iterator[Settings]:
    with MinioContainer() as minio:
        cfg = minio.get_config()
        yield Settings(
            s3_endpoint_url=f"http://{cfg['endpoint']}",
            s3_access_key=cfg["access_key"],
            s3_secret_key=cfg["secret_key"],
            s3_bucket="test-bucket",
        )


@pytest.fixture
async def docs_client(postgres_url: str, storage_settings: Settings) -> AsyncIterator[AsyncClient]:
    """API client wired to throwaway Postgres + MinIO, fresh schema per test."""
    engine = create_async_engine(postgres_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        await sync_rbac_catalog(session)

    storage = StorageService(storage_settings)
    await storage.ensure_bucket()

    async def override_get_db() -> AsyncIterator[AsyncSession]:
        async with factory() as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_storage_service] = lambda: storage
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
