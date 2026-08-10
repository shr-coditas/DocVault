"""Permission-filtered semantic search, end to end against real Postgres.

Documents are uploaded through the API and indexed through IndexingService, so
what these tests query is exactly what production writes. The embedder is the
deterministic ``FakeEmbedder``: a token-hash vector, which means "this document
is retrievable by this query" is an assertion and not a hope.

The authorization tests are the point of the file. Retrieval is the first place
where a visibility bug stops being a wrong HTTP status and starts being one
user's document text in another user's hands, so every arm of
``_accessible_condition`` is exercised here as well as in test_documents.py -
same rule, different call site, and AGENTS.md flags those two as twins.
"""

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings, get_settings
from app.db.base import Base
from app.db.session import get_db
from app.dependencies import get_embedder, get_reranker, get_storage_service
from app.main import create_app
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.repository.document_repository import DocumentRepository
from app.services.indexing_service import IndexingService
from app.services.search_service import MAX_LIMIT
from app.services.storage_service import StorageService
from tests.fakes import FakeEmbedder, FakeReranker
from tests.helpers import (
    WORKSPACES,
    add_member,
    create_schema,
    create_workspace,
    signup,
    sync_rbac_catalog,
)

pytestmark = pytest.mark.integration

# small chunks so a short document still produces several of them
SETTINGS = Settings(
    chunk_target_tokens=40, chunk_max_tokens=60, chunk_overlap_tokens=8, chunk_min_tokens=5
)

ME = "/api/v1/auth/me"


@dataclass
class Env:
    client: AsyncClient
    session_factory: async_sessionmaker[AsyncSession]
    service: IndexingService
    embedder: FakeEmbedder
    owner: dict[str, str]
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

    # one embedder for both sides: passages are indexed and queries are embedded
    # by the same object, which is exactly the invariant production relies on
    embedder = FakeEmbedder()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_storage_service] = lambda: test_storage
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_reranker] = FakeReranker

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        owner = await signup(client, "owner@example.com")
        workspace_id = await create_workspace(client, owner)
        yield Env(
            client=client,
            session_factory=factory,
            service=IndexingService(
                session_factory=factory,
                storage=test_storage,
                embedder=embedder,
                settings=SETTINGS,
            ),
            embedder=embedder,
            owner=owner,
            workspace_id=workspace_id,
        )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# -- helpers ---------------------------------------------------------------


def _prose(marker: str, sentences: int = 30) -> bytes:
    return " ".join(f"{marker} sentence {i} about the topic." for i in range(sentences)).encode()


async def _upload(
    env: Env,
    name: str,
    body: bytes,
    *,
    headers: dict[str, str] | None = None,
    workspace_id: str | None = None,
) -> str:
    response = await env.client.post(
        f"{WORKSPACES}/{workspace_id or env.workspace_id}/documents/upload",
        files={"file": (name, body, "text/plain")},
        headers=headers or env.owner,
    )
    assert response.status_code == 201, response.text
    document_id: str = response.json()["id"]
    return document_id


async def _index_all(env: Env) -> None:
    async with env.session_factory() as session:
        pending = await DocumentRepository(session).unindexed_ids(limit=100)
    for document_id in pending:
        outcome = await env.service.index_document(document_id)
        assert outcome.status == "indexed", outcome.detail


async def _seed(env: Env, marker: str, name: str | None = None) -> str:
    """Upload one document of ``marker``-flavoured prose and index it."""
    document_id = await _upload(env, name or f"{marker}.txt", _prose(marker))
    await _index_all(env)
    return document_id


async def _search(
    env: Env,
    query: str,
    *,
    headers: dict[str, str] | None = None,
    workspace_id: str | None = None,
    **params: object,
) -> dict:
    request_params = {"q": query, **params}
    request_params.setdefault("mode", "semantic")
    response = await env.client.get(
        f"{WORKSPACES}/{workspace_id or env.workspace_id}/search",
        params=request_params,
        headers=headers or env.owner,
    )
    assert response.status_code == 200, response.text
    body: dict = response.json()
    return body


async def _hit_documents(env: Env, query: str, **kwargs: object) -> set[str]:
    body = await _search(env, query, **kwargs)
    return {hit["document_id"] for hit in body["hits"]}


async def _user_id(env: Env, headers: dict[str, str]) -> str:
    response = await env.client.get(ME, headers=headers)
    assert response.status_code == 200, response.text
    user_id: str = response.json()["id"]
    return user_id


async def _set_restricted(
    env: Env, document_id: str, headers: dict[str, str] | None = None
) -> None:
    response = await env.client.put(
        f"{WORKSPACES}/{env.workspace_id}/documents/{document_id}/visibility",
        json={"visibility": "restricted"},
        headers=headers or env.owner,
    )
    assert response.status_code == 200, response.text


async def _member(env: Env, email: str, role: str) -> dict[str, str]:
    headers = await signup(env.client, email)
    response = await add_member(env.client, env.owner, env.workspace_id, email, role)
    assert response.status_code == 201, response.text
    return headers


# -- ranking ---------------------------------------------------------------


async def test_returns_chunks_of_the_matching_document_first(env: Env) -> None:
    football = await _seed(env, "football")
    await _upload(env, "cricket.txt", _prose("cricket"))
    await _index_all(env)

    body = await _search(env, "football", semantic_min_score=0.1)

    assert body["hits"], body
    assert {hit["document_id"] for hit in body["hits"]} == {football}
    # descending score, which is what "nearest first" has to mean to a caller
    scores = [hit["score"] for hit in body["hits"]]
    assert scores == sorted(scores, reverse=True)


async def test_every_accessible_chunk_is_scored_not_pre_filtered(env: Env) -> None:
    """The intuition worth unlearning: there is no topic pre-filter.

    Drop the threshold and the off-topic document's chunks come back too - they
    were always scored, they just lost. That is why a score floor exists at all.
    """
    football = await _seed(env, "football")
    cricket = await _upload(env, "cricket.txt", _prose("cricket"))
    await _index_all(env)

    body = await _search(env, "football", semantic_min_score=-1, limit=MAX_LIMIT)
    documents = [hit["document_id"] for hit in body["hits"]]

    assert set(documents) == {football, cricket}
    # ...and losing still means losing: every football chunk outranks every
    # cricket one
    best_cricket = next(index for index, doc in enumerate(documents) if doc == cricket)
    assert set(documents[:best_cricket]) == {football}


async def test_semantic_min_score_floor_excludes_the_off_topic_document(env: Env) -> None:
    football = await _seed(env, "football")
    await _upload(env, "cricket.txt", _prose("cricket"))
    await _index_all(env)

    assert await _hit_documents(env, "football", semantic_min_score=0.2) == {football}
    # a floor nothing can clear returns nothing rather than the best of a bad set
    assert await _hit_documents(env, "football", semantic_min_score=0.99) == set()


async def test_hits_carry_the_provenance_a_citation_needs(env: Env) -> None:
    document_id = await _seed(env, "rugby", name="Rugby Handbook.txt")

    body = await _search(env, "rugby", semantic_min_score=0.1)
    hit = body["hits"][0]

    assert hit["document_id"] == document_id
    assert hit["document_title"] == "Rugby Handbook.txt"
    assert hit["file_name"] == "Rugby Handbook.txt"
    assert uuid.UUID(hit["chunk_id"])
    assert hit["chunk_index"] >= 0
    assert "rugby" in hit["content"]
    assert hit["page_number"] is None  # txt is not paginated
    assert -1.0 <= hit["score"] <= 1.0


async def test_parameters_are_echoed_and_default_from_settings(env: Env) -> None:
    await _seed(env, "tennis")
    settings = get_settings()

    body = await _search(env, "tennis")

    assert body["query"] == "tennis"
    assert body["limit"] == settings.search_default_limit
    assert body["semantic_min_score"] == settings.search_semantic_min_score
    assert "min_score" not in body
    assert body["count"] == len(body["hits"])


async def test_limit_caps_the_hits(env: Env) -> None:
    await _seed(env, "hockey")

    body = await _search(env, "hockey", limit=2, semantic_min_score=0.1)

    assert len(body["hits"]) == 2
    assert body["limit"] == 2


async def test_public_default_is_hybrid_with_component_scores(env: Env) -> None:
    await _seed(env, "handbook")
    response = await env.client.get(
        f"{WORKSPACES}/{env.workspace_id}/search",
        params={"q": "handbook", "semantic_min_score": -1},
        headers=env.owner,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "hybrid"
    assert body["hits"]
    hit = body["hits"][0]
    assert hit["scores"]["fusion"] is not None
    assert hit["scores"]["rerank"] is not None
    assert hit["logical_key"]
    assert hit["index_generation"] == 1


async def test_lexical_mode_does_not_embed_the_query(env: Env) -> None:
    await _seed(env, "lexicalneedle")
    env.embedder.embedded_queries.clear()

    body = await _search(env, "lexicalneedle", mode="lexical")

    assert body["mode"] == "lexical"
    assert body["hits"]
    assert env.embedder.embedded_queries == []
    assert body["hits"][0]["scores"]["lexical"] is not None
    assert body["hits"][0]["scores"]["semantic"] is None


async def test_plural_document_scope_is_bounded_and_authoritative(env: Env) -> None:
    included = await _seed(env, "sharedword", name="included.txt")
    excluded = await _seed(env, "sharedword", name="excluded.txt")

    body = await _search(
        env, "sharedword", mode="hybrid", semantic_min_score=-1, document_ids=[included]
    )

    assert {hit["document_id"] for hit in body["hits"]} == {included}
    assert excluded not in {hit["document_id"] for hit in body["hits"]}


# -- what is searchable at all --------------------------------------------


async def test_unindexed_document_is_not_searchable(env: Env) -> None:
    await _upload(env, "pending.txt", _prose("badminton"))
    # deliberately not indexed
    assert await _hit_documents(env, "badminton", semantic_min_score=-1) == set()


async def test_trashed_document_drops_out_and_restore_brings_it_back(env: Env) -> None:
    document_id = await _seed(env, "cycling")
    assert await _hit_documents(env, "cycling", semantic_min_score=0.1) == {document_id}

    trashed = await env.client.delete(
        f"{WORKSPACES}/{env.workspace_id}/documents/{document_id}", headers=env.owner
    )
    assert trashed.status_code == 204
    assert await _hit_documents(env, "cycling", semantic_min_score=-1) == set()

    # chunks survive the trash, so a restore needs no re-index
    restored = await env.client.post(
        f"{WORKSPACES}/{env.workspace_id}/documents/{document_id}/restore", headers=env.owner
    )
    assert restored.status_code == 200
    assert await _hit_documents(env, "cycling", semantic_min_score=0.1) == {document_id}


async def test_superseded_generation_chunks_are_invisible(env: Env) -> None:
    """Chunks are visible only at the document's current index generation.

    Bumping the generation without rewriting chunks is not something the
    pipeline does today - the indexing transaction replaces both at once. It is
    simulated here because the guard exists for the step-wise workflow that will
    write chunks incrementally, and an unexercised guard is a guess.
    """
    document_id = await _seed(env, "climbing")
    assert await _hit_documents(env, "climbing", semantic_min_score=0.1) == {document_id}

    async with env.session_factory() as session:
        document = await session.get(Document, uuid.UUID(document_id))
        assert document is not None
        document.index_generation += 1
        await session.commit()

    assert await _hit_documents(env, "climbing", semantic_min_score=-1) == set()


async def test_same_dimension_chunks_from_an_inactive_profile_are_invisible(env: Env) -> None:
    document_id = await _seed(env, "profileisolated")
    async with env.session_factory() as session:
        await session.execute(
            update(DocumentChunk)
            .where(DocumentChunk.document_id == uuid.UUID(document_id))
            .values(embedding_profile_id="different-384-dimensional-profile")
        )
        await session.commit()

    assert await _hit_documents(env, "profileisolated", semantic_min_score=-1) == set()


async def test_permanent_delete_removes_the_chunks(env: Env) -> None:
    document_id = await _seed(env, "sailing")
    deleted = await env.client.delete(
        f"{WORKSPACES}/{env.workspace_id}/documents/{document_id}", headers=env.owner
    )
    assert deleted.status_code == 204
    purged = await env.client.delete(
        f"{WORKSPACES}/{env.workspace_id}/documents/{document_id}/permanent", headers=env.owner
    )
    assert purged.status_code == 204

    assert await _hit_documents(env, "sailing", semantic_min_score=-1) == set()


async def test_document_id_narrows_the_search_to_one_document(env: Env) -> None:
    first = await _upload(env, "one.txt", _prose("swimming"))
    second = await _upload(env, "two.txt", _prose("swimming"))
    await _index_all(env)

    assert await _hit_documents(env, "swimming", semantic_min_score=0.1) == {first, second}
    assert await _hit_documents(env, "swimming", semantic_min_score=0.1, document_id=first) == {
        first
    }


# -- authorization ---------------------------------------------------------


async def test_restricted_document_is_absent_for_another_member(env: Env) -> None:
    document_id = await _seed(env, "payroll")
    viewer = await _member(env, "viewer@example.com", "viewer")

    await _set_restricted(env, document_id)

    # the searcher is a viewer, not the workspace owner, so the SQL predicate is
    # what decides - not the owner override
    for mode in ("semantic", "lexical", "hybrid"):
        params = {"semantic_min_score": -1} if mode != "lexical" else {}
        assert await _hit_documents(env, "payroll", mode=mode, headers=viewer, **params) == set()
    # ...and the document owner still retrieves it
    assert await _hit_documents(env, "payroll", mode="hybrid", semantic_min_score=0.1) == {
        document_id
    }


async def test_user_grant_reveals_a_restricted_document_and_removal_hides_it(env: Env) -> None:
    document_id = await _seed(env, "payroll")
    viewer = await _member(env, "viewer@example.com", "viewer")
    viewer_id = await _user_id(env, viewer)
    await _set_restricted(env, document_id)

    granted = await env.client.post(
        f"{WORKSPACES}/{env.workspace_id}/documents/{document_id}/grants",
        json={"principal_type": "user", "principal_id": viewer_id},
        headers=env.owner,
    )
    assert granted.status_code == 201, granted.text
    assert await _hit_documents(env, "payroll", semantic_min_score=0.1, headers=viewer) == {
        document_id
    }

    revoked = await env.client.delete(
        f"{WORKSPACES}/{env.workspace_id}/documents/{document_id}/grants/user/{viewer_id}",
        headers=env.owner,
    )
    assert revoked.status_code == 204
    assert await _hit_documents(env, "payroll", semantic_min_score=-1, headers=viewer) == set()


async def test_team_grant_reveals_a_restricted_document(env: Env) -> None:
    document_id = await _seed(env, "payroll")
    viewer = await _member(env, "viewer@example.com", "viewer")
    viewer_id = await _user_id(env, viewer)
    await _set_restricted(env, document_id)

    team = await env.client.post(
        f"{WORKSPACES}/{env.workspace_id}/teams", json={"name": "Finance"}, headers=env.owner
    )
    assert team.status_code == 201, team.text
    team_id = team.json()["id"]
    added = await env.client.post(
        f"{WORKSPACES}/{env.workspace_id}/teams/{team_id}/members",
        json={"user_id": viewer_id},
        headers=env.owner,
    )
    assert added.status_code == 204, added.text

    assert await _hit_documents(env, "payroll", semantic_min_score=-1, headers=viewer) == set()

    granted = await env.client.post(
        f"{WORKSPACES}/{env.workspace_id}/documents/{document_id}/grants",
        json={"principal_type": "team", "principal_id": team_id},
        headers=env.owner,
    )
    assert granted.status_code == 201, granted.text
    assert await _hit_documents(env, "payroll", semantic_min_score=0.1, headers=viewer) == {
        document_id
    }


async def test_workspace_owner_retrieves_another_members_restricted_document(env: Env) -> None:
    editor = await _member(env, "editor@example.com", "editor")
    document_id = await _upload(env, "theirs.txt", _prose("appraisal"), headers=editor)
    await _index_all(env)
    await _set_restricted(env, document_id, headers=editor)

    # the workspace owner is unfiltered - the admin override, asserted here
    # because search must not be a way around it either
    assert await _hit_documents(env, "appraisal", semantic_min_score=0.1) == {document_id}


async def test_another_workspace_is_never_searched(env: Env) -> None:
    mine = await _seed(env, "logistics")
    other = await create_workspace(env.client, env.owner, name="Other")
    theirs = await _upload(env, "theirs.txt", _prose("logistics"), workspace_id=other)
    await _index_all(env)

    assert await _hit_documents(env, "logistics", semantic_min_score=0.1) == {mine}
    assert await _hit_documents(env, "logistics", semantic_min_score=0.1, workspace_id=other) == {
        theirs
    }


async def test_non_member_gets_404(env: Env) -> None:
    await _seed(env, "logistics")
    outsider = await signup(env.client, "outsider@example.com")

    response = await env.client.get(
        f"{WORKSPACES}/{env.workspace_id}/search",
        params={"q": "logistics"},
        headers=outsider,
    )
    # 404, not 403: a non-member must not learn the workspace exists
    assert response.status_code == 404


async def test_unauthenticated_request_is_rejected(env: Env) -> None:
    response = await env.client.get(
        f"{WORKSPACES}/{env.workspace_id}/search", params={"q": "logistics"}
    )
    assert response.status_code == 401


# -- request validation ----------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        {"q": ""},  # empty query
        {"q": "x", "mode": "unknown"},
        {"q": "x", "limit": 0},
        {"q": "x", "limit": MAX_LIMIT + 1},
        {"q": "x", "semantic_min_score": 1.5},
        {"q": "x", "min_score": 0.1},
        {"q": "x", "mode": "lexical", "semantic_min_score": 0.1},
        {
            "q": "x",
            "document_id": str(uuid.uuid4()),
            "document_ids": [str(uuid.uuid4())],
        },
        {"q": "x", "document_id": "not-a-uuid"},
    ],
)
async def test_bad_parameters_are_rejected(env: Env, params: dict[str, object]) -> None:
    response = await env.client.get(
        f"{WORKSPACES}/{env.workspace_id}/search", params=params, headers=env.owner
    )
    assert response.status_code == 422, response.text


async def test_whitespace_query_returns_nothing_without_embedding_it(env: Env) -> None:
    await _seed(env, "rowing")
    body = await _search(env, "   ", semantic_min_score=-1)
    assert body["hits"] == []
    assert body["count"] == 0
