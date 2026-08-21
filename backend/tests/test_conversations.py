"""Creator-owned conversation lifecycle API."""

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from uuid6 import uuid7

from app.db.base import Base
from app.db.session import get_db
from app.main import create_app
from app.models.conversation import (
    ConversationMessage,
    MessageKind,
    MessageRole,
    MessageStatus,
)
from app.models.document import Document, DocumentVisibility
from app.repository.conversation_repository import ConversationRepository
from tests.helpers import (
    WORKSPACES,
    add_member,
    create_schema,
    create_workspace,
    signup,
    sync_rbac_catalog,
)

pytestmark = pytest.mark.integration


@dataclass(frozen=True, slots=True)
class Env:
    client: AsyncClient
    app: FastAPI
    session_factory: async_sessionmaker[AsyncSession]
    owner: dict[str, str]
    viewer: dict[str, str]
    outsider: dict[str, str]
    owner_id: uuid.UUID
    viewer_id: uuid.UUID
    workspace_id: uuid.UUID
    other_workspace_id: uuid.UUID
    visible_document_id: uuid.UUID
    restricted_document_id: uuid.UUID
    trashed_document_id: uuid.UUID
    other_workspace_document_id: uuid.UUID

    @property
    def conversations_url(self) -> str:
        return f"{WORKSPACES}/{self.workspace_id}/conversations"


def _document(
    *,
    workspace_id: uuid.UUID,
    owner_id: uuid.UUID,
    marker: str,
    visibility: DocumentVisibility = DocumentVisibility.WORKSPACE,
    deleted_at: datetime | None = None,
) -> Document:
    return Document(
        id=uuid7(),
        workspace_id=workspace_id,
        folder_id=None,
        owner_id=owner_id,
        title=f"{marker} title",
        file_name=f"{marker}.md",
        mime_type="text/markdown",
        size_bytes=10,
        storage_key=f"tests/{uuid7()}/{marker}.md",
        visibility=visibility,
        deleted_at=deleted_at,
    )


@pytest.fixture
async def env(postgres_url: str) -> AsyncIterator[Env]:
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
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        owner = await signup(client, "conversation-owner@example.com")
        viewer = await signup(client, "conversation-viewer@example.com")
        outsider = await signup(client, "conversation-outsider@example.com")
        owner_id = uuid.UUID((await client.get("/api/v1/auth/me", headers=owner)).json()["id"])
        viewer_id = uuid.UUID((await client.get("/api/v1/auth/me", headers=viewer)).json()["id"])
        workspace_id = uuid.UUID(await create_workspace(client, owner, "Conversation workspace"))
        other_workspace_id = uuid.UUID(await create_workspace(client, owner, "Other workspace"))
        added = await add_member(
            client,
            owner,
            str(workspace_id),
            "conversation-viewer@example.com",
            "viewer",
        )
        assert added.status_code == 201, added.text

        visible = _document(
            workspace_id=workspace_id,
            owner_id=owner_id,
            marker="visible",
        )
        restricted = _document(
            workspace_id=workspace_id,
            owner_id=owner_id,
            marker="restricted",
            visibility=DocumentVisibility.RESTRICTED,
        )
        trashed = _document(
            workspace_id=workspace_id,
            owner_id=owner_id,
            marker="trashed",
            deleted_at=datetime.now(UTC),
        )
        other_workspace_document = _document(
            workspace_id=other_workspace_id,
            owner_id=owner_id,
            marker="other-workspace",
        )
        async with factory() as session:
            session.add_all([visible, restricted, trashed, other_workspace_document])
            await session.commit()

        yield Env(
            client=client,
            app=app,
            session_factory=factory,
            owner=owner,
            viewer=viewer,
            outsider=outsider,
            owner_id=owner_id,
            viewer_id=viewer_id,
            workspace_id=workspace_id,
            other_workspace_id=other_workspace_id,
            visible_document_id=visible.id,
            restricted_document_id=restricted.id,
            trashed_document_id=trashed.id,
            other_workspace_document_id=other_workspace_document.id,
        )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def _create(
    env: Env,
    *,
    headers: dict[str, str] | None = None,
    scope_mode: str = "workspace",
    document_ids: list[uuid.UUID] | None = None,
):
    body: dict[str, object] = {"scope_mode": scope_mode}
    if document_ids is not None:
        body["document_ids"] = [str(document_id) for document_id in document_ids]
    return await env.client.post(
        env.conversations_url,
        json=body,
        headers=headers or env.owner,
    )


async def test_create_workspace_and_selected_conversations_with_frozen_scope(env: Env) -> None:
    workspace = await _create(env)
    assert workspace.status_code == 201, workspace.text
    assert workspace.json()["scope_mode"] == "workspace"
    assert workspace.json()["documents"] == []
    assert workspace.json()["title"] == "New conversation"

    selected = await _create(
        env,
        scope_mode="selected",
        document_ids=[env.restricted_document_id, env.visible_document_id],
    )
    assert selected.status_code == 201, selected.text
    assert [row["document_id"] for row in selected.json()["documents"]] == [
        str(env.restricted_document_id),
        str(env.visible_document_id),
    ]
    assert [row["position"] for row in selected.json()["documents"]] == [0, 1]
    assert selected.json()["documents"][0]["title"] == "restricted title"
    assert selected.json()["documents"][0]["file_name"] == "restricted.md"

    audit = await env.client.get(
        f"{WORKSPACES}/{env.workspace_id}/audit",
        headers=env.owner,
    )
    created = [row for row in audit.json() if row["action"] == "conversation.created"]
    assert len(created) == 2
    assert created[0]["extra"].keys() == {"scope_mode", "document_count"}


@pytest.mark.parametrize(
    "body",
    [
        {"scope_mode": "selected"},
        {"scope_mode": "workspace", "document_ids": [str(uuid.UUID(int=1))]},
        {
            "scope_mode": "selected",
            "document_ids": [str(uuid.UUID(int=1)), str(uuid.UUID(int=1))],
        },
        {"scope_mode": "workspace", "unexpected": True},
    ],
)
async def test_create_rejects_invalid_scope_shapes(env: Env, body: dict[str, object]) -> None:
    response = await env.client.post(env.conversations_url, json=body, headers=env.owner)
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")


async def test_selected_scope_hides_unavailable_documents(env: Env) -> None:
    for document_id in (
        uuid7(),
        env.trashed_document_id,
        env.other_workspace_document_id,
    ):
        response = await _create(
            env,
            scope_mode="selected",
            document_ids=[document_id],
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "one or more documents not found"

    restricted = await _create(
        env,
        headers=env.viewer,
        scope_mode="selected",
        document_ids=[env.restricted_document_id],
    )
    assert restricted.status_code == 404

    visible = await _create(
        env,
        headers=env.viewer,
        scope_mode="selected",
        document_ids=[env.visible_document_id],
    )
    assert visible.status_code == 201


async def test_conversations_are_creator_private_even_from_workspace_owner(env: Env) -> None:
    viewer_conversation = await _create(env, headers=env.viewer)
    assert viewer_conversation.status_code == 201
    conversation_id = viewer_conversation.json()["id"]

    owner_get = await env.client.get(
        f"{env.conversations_url}/{conversation_id}", headers=env.owner
    )
    assert owner_get.status_code == 404
    assert owner_get.headers["content-type"].startswith("application/problem+json")

    wrong_workspace = await env.client.get(
        f"{WORKSPACES}/{env.other_workspace_id}/conversations/{conversation_id}",
        headers=env.viewer,
    )
    assert wrong_workspace.status_code == 404

    nonmember = await env.client.get(env.conversations_url, headers=env.outsider)
    assert nonmember.status_code == 404


async def test_conversation_list_uses_an_opaque_creator_scoped_cursor(env: Env) -> None:
    ids = []
    for _ in range(3):
        response = await _create(env)
        ids.append(response.json()["id"])
    await _create(env, headers=env.viewer)

    first = await env.client.get(
        env.conversations_url,
        params={"limit": 2},
        headers=env.owner,
    )
    assert first.status_code == 200
    first_body = first.json()
    assert len(first_body["items"]) == 2
    assert first_body["next_cursor"]
    assert {item["id"] for item in first_body["items"]}.issubset(set(ids))

    second = await env.client.get(
        env.conversations_url,
        params={"limit": 2, "cursor": first_body["next_cursor"]},
        headers=env.owner,
    )
    second_body = second.json()
    assert len(second_body["items"]) == 1
    assert second_body["next_cursor"] is None
    assert {item["id"] for item in [*first_body["items"], *second_body["items"]]} == set(ids)

    invalid = await env.client.get(
        env.conversations_url,
        params={"cursor": "not-a-cursor"},
        headers=env.owner,
    )
    assert invalid.status_code == 422
    assert invalid.json()["detail"] == "invalid conversation cursor"


async def test_message_listing_is_sequence_paginated_and_creator_owned(env: Env) -> None:
    created = await _create(env)
    conversation_id = uuid.UUID(created.json()["id"])
    async with env.session_factory() as session:
        repository = ConversationRepository(session)
        messages = []
        for turn_number in range(2):
            turn_id = uuid7()
            messages.extend(
                [
                    ConversationMessage(
                        id=uuid7(),
                        conversation_id=conversation_id,
                        turn_id=turn_id,
                        sequence=turn_number * 2 + 1,
                        role=MessageRole.USER,
                        status=MessageStatus.COMPLETE,
                        kind=None,
                        content=f"Question {turn_number + 1}",
                        client_message_id=uuid7(),
                    ),
                    ConversationMessage(
                        id=uuid7(),
                        conversation_id=conversation_id,
                        turn_id=turn_id,
                        sequence=turn_number * 2 + 2,
                        role=MessageRole.ASSISTANT,
                        status=MessageStatus.COMPLETE,
                        kind=MessageKind.NO_SOURCES,
                        content="No sources found.",
                        client_message_id=None,
                    ),
                ]
            )
        repository.add_messages(messages)
        await session.commit()

    url = f"{env.conversations_url}/{conversation_id}/messages"
    first = await env.client.get(url, params={"limit": 2}, headers=env.owner)
    assert [item["sequence"] for item in first.json()["items"]] == [1, 2]
    assert first.json()["next_after_sequence"] == 2

    second = await env.client.get(
        url,
        params={"after_sequence": 2, "limit": 2},
        headers=env.owner,
    )
    assert [item["sequence"] for item in second.json()["items"]] == [3, 4]
    assert second.json()["next_after_sequence"] is None
    assert "resolved_query" not in second.json()["items"][0]
    assert "lease_token" not in second.json()["items"][0]

    hidden = await env.client.get(url, headers=env.viewer)
    assert hidden.status_code == 404


async def test_delete_is_hard_creator_owned_and_content_free_in_audit(env: Env) -> None:
    created = await _create(env)
    conversation_id = created.json()["id"]
    url = f"{env.conversations_url}/{conversation_id}"

    forbidden_by_hiding = await env.client.delete(url, headers=env.viewer)
    assert forbidden_by_hiding.status_code == 404

    deleted = await env.client.delete(url, headers=env.owner)
    assert deleted.status_code == 204
    assert (await env.client.get(url, headers=env.owner)).status_code == 404

    audit = await env.client.get(
        f"{WORKSPACES}/{env.workspace_id}/audit",
        headers=env.owner,
    )
    deletion = next(row for row in audit.json() if row["action"] == "conversation.deleted")
    assert deletion["resource_id"] == conversation_id
    assert deletion["extra"] == {}
