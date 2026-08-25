"""Persistence invariants for creator-owned conversation history."""

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from uuid6 import uuid7

from app.db.base import Base
from app.models.conversation import (
    Conversation,
    ConversationDocument,
    ConversationMessage,
    ConversationScope,
    MessageKind,
    MessageRole,
    MessageSource,
    MessageStatus,
)
from app.models.document import Document, DocumentVisibility
from app.models.user import User
from app.models.workspace import Workspace
from app.repository.conversation_repository import ConversationRepository
from app.repository.document_repository import DocumentRepository
from tests.helpers import create_schema

pytestmark = pytest.mark.integration


@dataclass(frozen=True, slots=True)
class Seeded:
    owner: User
    other_user: User
    workspace: Workspace
    other_workspace: Workspace
    visible: Document
    restricted: Document
    trashed: Document
    other_workspace_document: Document


@pytest.fixture
async def session(postgres_url: str) -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(postgres_url)
    await create_schema(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as db:
        yield db
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


def _user(email: str) -> User:
    return User(
        id=uuid7(),
        email=email,
        hashed_password="not-used",
        full_name=email.split("@", maxsplit=1)[0],
    )


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
        file_name=f"{marker}.txt",
        mime_type="text/plain",
        size_bytes=10,
        storage_key=f"tests/{uuid7()}/{marker}.txt",
        visibility=visibility,
        deleted_at=deleted_at,
        indexed=False,
    )


async def _seed(session: AsyncSession) -> Seeded:
    owner = _user("owner@example.com")
    other_user = _user("other@example.com")
    session.add_all([owner, other_user])
    await session.flush()

    workspace = Workspace(id=uuid7(), name="Primary", description="", created_by=owner.id)
    other_workspace = Workspace(id=uuid7(), name="Other", description="", created_by=other_user.id)
    session.add_all([workspace, other_workspace])
    await session.flush()

    visible = _document(
        workspace_id=workspace.id,
        owner_id=owner.id,
        marker="visible",
    )
    restricted = _document(
        workspace_id=workspace.id,
        owner_id=owner.id,
        marker="restricted",
        visibility=DocumentVisibility.RESTRICTED,
    )
    trashed = _document(
        workspace_id=workspace.id,
        owner_id=owner.id,
        marker="trashed",
        deleted_at=datetime.now(UTC),
    )
    other_workspace_document = _document(
        workspace_id=other_workspace.id,
        owner_id=other_user.id,
        marker="other-workspace",
    )
    session.add_all([visible, restricted, trashed, other_workspace_document])
    await session.commit()
    return Seeded(
        owner=owner,
        other_user=other_user,
        workspace=workspace,
        other_workspace=other_workspace,
        visible=visible,
        restricted=restricted,
        trashed=trashed,
        other_workspace_document=other_workspace_document,
    )


def _conversation(
    seeded: Seeded,
    *,
    creator: User | None = None,
    workspace: Workspace | None = None,
    title: str = "Conversation",
    updated_at: datetime | None = None,
) -> Conversation:
    return Conversation(
        id=uuid7(),
        workspace_id=(workspace or seeded.workspace).id,
        created_by=(creator or seeded.owner).id,
        title=title,
        scope_mode=ConversationScope.SELECTED,
        updated_at=updated_at or datetime.now(UTC),
    )


def _complete_turn(
    conversation_id: uuid.UUID,
) -> tuple[ConversationMessage, ConversationMessage]:
    user = ConversationMessage(
        id=uuid7(),
        conversation_id=conversation_id,
        sequence=1,
        role=MessageRole.USER,
        status=MessageStatus.COMPLETE,
        kind=None,
        content="What does the document say?",
    )
    assistant = ConversationMessage(
        id=uuid7(),
        conversation_id=conversation_id,
        sequence=2,
        role=MessageRole.ASSISTANT,
        status=MessageStatus.COMPLETE,
        kind=MessageKind.ANSWER,
        content="A grounded answer [1].",
    )
    return user, assistant


async def test_batched_access_filters_workspace_deletion_and_visibility(
    session: AsyncSession,
) -> None:
    seeded = await _seed(session)
    candidates = [
        seeded.visible.id,
        seeded.restricted.id,
        seeded.trashed.id,
        seeded.other_workspace_document.id,
        uuid7(),
        seeded.visible.id,
    ]
    repository = DocumentRepository(session)

    viewer_rows = await repository.accessible_active_by_ids(
        seeded.workspace.id,
        candidates,
        access=(seeded.other_user.id, []),
    )
    assert {document.id for document in viewer_rows} == {seeded.visible.id}

    owner_rows = await repository.accessible_active_by_ids(
        seeded.workspace.id,
        candidates,
        access=None,
    )
    assert {document.id for document in owner_rows} == {
        seeded.visible.id,
        seeded.restricted.id,
    }


async def test_repository_hides_other_creators_and_workspaces_and_pages_by_cursor(
    session: AsyncSession,
) -> None:
    seeded = await _seed(session)
    now = datetime.now(UTC)
    newest = _conversation(seeded, title="Newest", updated_at=now)
    older = _conversation(seeded, title="Older", updated_at=now - timedelta(minutes=1))
    other_creator = _conversation(seeded, creator=seeded.other_user, title="Private")
    other_workspace = _conversation(
        seeded,
        workspace=seeded.other_workspace,
        creator=seeded.other_user,
        title="Elsewhere",
    )
    repository = ConversationRepository(session)
    for conversation in (newest, older, other_creator, other_workspace):
        repository.add(conversation)
    await session.commit()

    page = await repository.list_owned(
        workspace_id=seeded.workspace.id,
        creator_id=seeded.owner.id,
        limit=1,
    )
    assert [conversation.id for conversation in page] == [newest.id]

    next_page = await repository.list_owned(
        workspace_id=seeded.workspace.id,
        creator_id=seeded.owner.id,
        limit=10,
        before=(newest.updated_at, newest.id),
    )
    assert [conversation.id for conversation in next_page] == [older.id]
    assert (
        await repository.get_owned(
            newest.id,
            workspace_id=seeded.workspace.id,
            creator_id=seeded.other_user.id,
        )
        is None
    )
    assert (
        await repository.get_owned(
            newest.id,
            workspace_id=seeded.other_workspace.id,
            creator_id=seeded.owner.id,
        )
        is None
    )


async def test_conversation_cascades_but_document_history_does_not(
    session: AsyncSession,
) -> None:
    seeded = await _seed(session)
    conversation = _conversation(seeded)
    repository = ConversationRepository(session)
    repository.add(conversation)
    repository.add_documents(
        [
            ConversationDocument(
                conversation_id=conversation.id,
                document_id=seeded.visible.id,
                position=0,
                title_snapshot=seeded.visible.title,
                file_name_snapshot=seeded.visible.file_name,
            )
        ]
    )
    user, assistant = _complete_turn(conversation.id)
    repository.add_messages([user, assistant])
    source = MessageSource(
        id=uuid7(),
        message_id=assistant.id,
        document_id=seeded.visible.id,
        chunk_id=uuid7(),
        document_title_snapshot=seeded.visible.title,
        section_path="Policy",
        chunk_type="paragraph",
        retrieval_rank=1,
        supplied_to_model=True,
        citation_marker=1,
    )
    repository.add_sources([source])
    await session.commit()

    await session.delete(seeded.visible)
    await session.commit()
    assert await session.get(ConversationDocument, (conversation.id, seeded.visible.id))
    assert await session.get(MessageSource, source.id)

    await repository.delete(conversation)
    await session.commit()
    conversation_documents = await session.scalar(
        select(func.count())
        .select_from(ConversationDocument)
        .where(ConversationDocument.conversation_id == conversation.id)
    )
    messages = await session.scalar(
        select(func.count())
        .select_from(ConversationMessage)
        .where(ConversationMessage.conversation_id == conversation.id)
    )
    sources = await session.scalar(
        select(func.count()).select_from(MessageSource).where(MessageSource.id == source.id)
    )
    assert conversation_documents == 0
    assert messages == 0
    assert sources == 0


async def test_pending_assistant_is_not_a_valid_persisted_message(
    session: AsyncSession,
) -> None:
    seeded = await _seed(session)
    conversation = _conversation(seeded)
    session.add(conversation)
    await session.commit()
    conversation_id = conversation.id

    invalid = ConversationMessage(
        id=uuid7(),
        conversation_id=conversation_id,
        sequence=1,
        role=MessageRole.ASSISTANT,
        status=MessageStatus.PENDING,
        kind=None,
        content=None,
    )
    session.add(invalid)
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


async def test_citation_requires_a_source_supplied_to_the_model(
    session: AsyncSession,
) -> None:
    seeded = await _seed(session)
    conversation = _conversation(seeded)
    session.add(conversation)
    _, assistant = _complete_turn(conversation.id)
    session.add(assistant)
    session.add(
        MessageSource(
            id=uuid7(),
            message_id=assistant.id,
            document_id=seeded.visible.id,
            chunk_id=uuid7(),
            document_title_snapshot=seeded.visible.title,
            section_path=None,
            chunk_type="paragraph",
            retrieval_rank=1,
            supplied_to_model=False,
            citation_marker=1,
        )
    )

    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()
