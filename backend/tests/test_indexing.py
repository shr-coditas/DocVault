"""The indexing pipeline, end to end against real Postgres + MinIO.

Documents are uploaded through the API so the storage keys, checksums and rows
are exactly what production would produce; indexing is then driven directly
through IndexingService rather than by shelling out to the script.
"""

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings
from app.db.base import Base
from app.db.session import get_db
from app.dependencies import get_storage_service
from app.main import create_app
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.repository.document_index_repository import DocumentIndexRepository
from app.repository.document_repository import DocumentRepository
from app.services.indexing_service import IndexingService
from app.services.storage_service import StorageService
from tests.fakes import ExplodingEmbedder, FakeEmbedder
from tests.helpers import WORKSPACES, create_schema, create_workspace, signup, sync_rbac_catalog

pytestmark = pytest.mark.integration

SETTINGS = Settings(
    chunk_target_tokens=40, chunk_max_tokens=60, chunk_overlap_tokens=8, chunk_min_tokens=5
)


@dataclass
class Env:
    client: AsyncClient
    session_factory: async_sessionmaker[AsyncSession]
    storage: StorageService
    embedder: FakeEmbedder
    service: IndexingService
    headers: dict[str, str]
    workspace_id: str


@pytest.fixture
async def env(postgres_url: str, test_storage: StorageService) -> AsyncIterator[Env]:
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

    embedder = FakeEmbedder()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        headers = await signup(client, "indexer@example.com")
        workspace_id = await create_workspace(client, headers)
        yield Env(
            client=client,
            session_factory=factory,
            storage=test_storage,
            embedder=embedder,
            service=IndexingService(
                session_factory=factory,
                storage=test_storage,
                embedder=embedder,
                settings=SETTINGS,
            ),
            headers=headers,
            workspace_id=workspace_id,
        )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def _upload(env: Env, name: str, body: bytes, mime: str = "text/plain") -> str:
    response = await env.client.post(
        f"{WORKSPACES}/{env.workspace_id}/documents/upload",
        files={"file": (name, body, mime)},
        headers=env.headers,
    )
    assert response.status_code == 201, response.text
    document_id: str = response.json()["id"]
    return document_id


def _prose(marker: str, sentences: int = 40) -> bytes:
    return " ".join(f"{marker} sentence {i} about the topic." for i in range(sentences)).encode()


async def _index_all(env: Env, **kwargs: object) -> list[str]:
    async with env.session_factory() as session:
        pending = await DocumentRepository(session).unindexed_ids(limit=50, **kwargs)  # type: ignore[arg-type]
    statuses = []
    for document_id in pending:
        outcome = await env.service.index_document(document_id)
        statuses.append(outcome.status)
    return statuses


async def _document(env: Env, document_id: str) -> Document:
    async with env.session_factory() as session:
        document = await DocumentRepository(session).get(uuid.UUID(document_id))
        assert document is not None
        return document


async def _chunks(env: Env, document_id: str) -> list[DocumentChunk]:
    async with env.session_factory() as session:
        stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == uuid.UUID(document_id))
            .order_by(DocumentChunk.chunk_index)
        )
        return list((await session.execute(stmt)).scalars())


async def _all_chunks(env: Env, document_id: str) -> list[DocumentChunk]:
    async with env.session_factory() as session:
        stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == uuid.UUID(document_id))
            .order_by(DocumentChunk.chunk_index)
        )
        return list((await session.execute(stmt)).scalars())


# -- the happy path --------------------------------------------------------


async def test_indexes_pending_documents(env: Env) -> None:
    first = await _upload(env, "a.txt", _prose("alpha"))
    second = await _upload(env, "b.txt", _prose("beta"))

    assert await _index_all(env) == ["indexed", "indexed"]

    for document_id in (first, second):
        document = await _document(env, document_id)
        assert document.indexed is True
        assert document.indexed_at is not None
        assert document.index_error is None
        assert await _chunks(env, document_id)


async def test_chunks_carry_tenant_and_model_metadata(env: Env) -> None:
    document_id = await _upload(env, "a.txt", _prose("gamma"))
    await _index_all(env)

    chunks = await _chunks(env, document_id)
    assert chunks
    for chunk in chunks:
        # denormalised so the tenant filter stands without the join
        assert str(chunk.workspace_id) == env.workspace_id
        assert chunk.embedding_model == "fake-token-hash"
        assert len(chunk.embedding) == 384
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


# -- idempotency -----------------------------------------------------------


async def test_rerun_processes_nothing(env: Env) -> None:
    document_id = await _upload(env, "a.txt", _prose("delta"))
    await _index_all(env)
    before = len(await _chunks(env, document_id))

    # the selector only sees indexed = false, so a second run has no work
    assert await _index_all(env) == []
    assert len(await _chunks(env, document_id)) == before
    assert (await _document(env, document_id)).indexed is True


async def test_title_change_is_display_only_and_does_not_schedule_reindex(env: Env) -> None:
    document_id = await _upload(env, "a.txt", _prose("display title"))
    await _index_all(env)

    response = await env.client.patch(
        f"{WORKSPACES}/{env.workspace_id}/documents/{document_id}",
        json={"title": "Renamed for display"},
        headers=env.headers,
    )

    assert response.status_code == 200
    assert response.json()["title"] == "Renamed for display"
    assert (await _document(env, document_id)).indexed is True
    assert await _index_all(env) == []


# -- failure handling ------------------------------------------------------


async def test_unreadable_file_records_error_and_run_continues(env: Env) -> None:
    broken = await _upload(env, "broken.pdf", b"%PDF-1.4 not really a pdf", "application/pdf")
    healthy = await _upload(env, "ok.txt", _prose("zeta"))

    statuses = await _index_all(env)
    assert sorted(statuses) == ["failed", "indexed"]

    failed = await _document(env, broken)
    assert failed.indexed is False
    assert failed.index_error is not None
    assert await _chunks(env, broken) == []

    # one bad document must not stop the run
    assert (await _document(env, healthy)).indexed is True


async def test_file_with_no_text_is_recorded_as_failed(env: Env) -> None:
    empty = await _upload(env, "empty.txt", b"   \n  \n ")
    assert await _index_all(env) == ["failed"]

    document = await _document(env, empty)
    assert document.indexed is False
    assert document.index_error == "no extractable text"


async def test_embedding_failure_is_recorded_not_raised(env: Env) -> None:
    document_id = await _upload(env, "a.txt", _prose("eta"))
    env.service.embedder = ExplodingEmbedder()

    assert await _index_all(env) == ["failed"]
    document = await _document(env, document_id)
    assert document.indexed is False
    assert "embedding provider unavailable" in (document.index_error or "")
    assert await _chunks(env, document_id) == []


# -- selection -------------------------------------------------------------


async def test_trashed_documents_are_not_selected(env: Env) -> None:
    document_id = await _upload(env, "a.txt", _prose("theta"))
    response = await env.client.delete(
        f"{WORKSPACES}/{env.workspace_id}/documents/{document_id}", headers=env.headers
    )
    assert response.status_code == 204

    assert await _index_all(env) == []
    assert (await _document(env, document_id)).indexed is False


async def test_restored_document_is_picked_up(env: Env) -> None:
    document_id = await _upload(env, "a.txt", _prose("iota"))
    await env.client.delete(
        f"{WORKSPACES}/{env.workspace_id}/documents/{document_id}", headers=env.headers
    )
    await env.client.post(
        f"{WORKSPACES}/{env.workspace_id}/documents/{document_id}/restore", headers=env.headers
    )

    assert await _index_all(env) == ["indexed"]


async def test_workspace_scoping(env: Env) -> None:
    mine = await _upload(env, "mine.txt", _prose("kappa"))
    other_workspace = await create_workspace(env.client, env.headers, name="Other")
    response = await env.client.post(
        f"{WORKSPACES}/{other_workspace}/documents/upload",
        files={"file": ("theirs.txt", _prose("lambda"), "text/plain")},
        headers=env.headers,
    )
    assert response.status_code == 201
    theirs = response.json()["id"]

    assert await _index_all(env, workspace_id=uuid.UUID(env.workspace_id)) == ["indexed"]
    assert (await _document(env, mine)).indexed is True
    assert (await _document(env, theirs)).indexed is False


# -- claim semantics -------------------------------------------------------


async def test_claiming_twice_skips(env: Env) -> None:
    document_id = await _upload(env, "a.txt", _prose("mu"))
    await _index_all(env)

    # already indexed, so the claim finds nothing
    outcome = await env.service.index_document(uuid.UUID(document_id))
    assert outcome.status == "skipped"


async def test_unexpired_lease_blocks_a_competitor_and_expiry_allows_recovery(env: Env) -> None:
    document_id = uuid.UUID(await _upload(env, "lease.txt", _prose("lease")))

    first = await env.service.claim(document_id)
    assert first is not None
    assert await env.service.claim(document_id) is None

    async with env.session_factory() as session:
        run = await DocumentIndexRepository(session).get(first.run_id)
        assert run is not None
        run.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        await session.commit()

    recovered = await env.service.claim(document_id)
    assert recovered is not None
    assert recovered.run_id == first.run_id
    assert recovered.lease_token != first.lease_token


async def test_audit_row_is_written_for_a_successful_index(env: Env) -> None:
    await _upload(env, "a.txt", _prose("nu"))
    await _index_all(env)

    response = await env.client.get(f"{WORKSPACES}/{env.workspace_id}/audit", headers=env.headers)
    actions = [row["action"] for row in response.json()]
    assert "document.indexed" in actions
