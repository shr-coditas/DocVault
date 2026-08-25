"""All SQL for creator-owned conversation history."""

import uuid
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.models.conversation import (
    Conversation,
    ConversationDocument,
    ConversationMessage,
    MessageSource,
)


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def add(self, conversation: Conversation) -> None:
        self.session.add(conversation)

    def add_documents(self, documents: Sequence[ConversationDocument]) -> None:
        self.session.add_all(documents)

    def add_messages(self, messages: Sequence[ConversationMessage]) -> None:
        self.session.add_all(messages)

    def add_sources(self, sources: Sequence[MessageSource]) -> None:
        self.session.add_all(sources)

    async def delete(self, conversation: Conversation) -> None:
        await self.session.delete(conversation)

    async def get_owned(
        self,
        conversation_id: uuid.UUID,
        *,
        workspace_id: uuid.UUID,
        creator_id: uuid.UUID,
    ) -> Conversation | None:
        stmt = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.workspace_id == workspace_id,
            Conversation.created_by == creator_id,
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_owned_for_update(
        self,
        conversation_id: uuid.UUID,
        *,
        workspace_id: uuid.UUID,
        creator_id: uuid.UUID,
    ) -> Conversation | None:
        stmt = (
            select(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.workspace_id == workspace_id,
                Conversation.created_by == creator_id,
            )
            .with_for_update()
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list_owned(
        self,
        *,
        workspace_id: uuid.UUID,
        creator_id: uuid.UUID,
        limit: int,
        before: tuple[datetime, uuid.UUID] | None = None,
    ) -> list[Conversation]:
        stmt = select(Conversation).where(
            Conversation.workspace_id == workspace_id,
            Conversation.created_by == creator_id,
        )
        if before is not None:
            updated_at, conversation_id = before
            stmt = stmt.where(
                or_(
                    Conversation.updated_at < updated_at,
                    and_(
                        Conversation.updated_at == updated_at,
                        Conversation.id < conversation_id,
                    ),
                )
            )
        stmt = stmt.order_by(Conversation.updated_at.desc(), Conversation.id.desc()).limit(limit)
        return list((await self.session.execute(stmt)).scalars())

    async def list_documents(self, conversation_id: uuid.UUID) -> list[ConversationDocument]:
        stmt = (
            select(ConversationDocument)
            .where(ConversationDocument.conversation_id == conversation_id)
            .order_by(ConversationDocument.position)
        )
        return list((await self.session.execute(stmt)).scalars())

    async def documents_for_conversations(
        self, conversation_ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, list[ConversationDocument]]:
        ids = list(dict.fromkeys(conversation_ids))
        if not ids:
            return {}
        stmt = (
            select(ConversationDocument)
            .where(ConversationDocument.conversation_id.in_(ids))
            .order_by(ConversationDocument.conversation_id, ConversationDocument.position)
        )
        grouped: dict[uuid.UUID, list[ConversationDocument]] = {
            conversation_id: [] for conversation_id in ids
        }
        for document in (await self.session.execute(stmt)).scalars():
            grouped[document.conversation_id].append(document)
        return grouped

    async def list_messages_after(
        self,
        conversation_id: uuid.UUID,
        *,
        after_sequence: int = 0,
        limit: int,
    ) -> list[ConversationMessage]:
        stmt = (
            select(ConversationMessage)
            .where(
                ConversationMessage.conversation_id == conversation_id,
                ConversationMessage.sequence > after_sequence,
            )
            .order_by(ConversationMessage.sequence)
            .limit(limit)
        )
        return list((await self.session.execute(stmt)).scalars())

    async def next_sequence(self, conversation_id: uuid.UUID) -> int:
        stmt = select(func.coalesce(func.max(ConversationMessage.sequence), 0)).where(
            ConversationMessage.conversation_id == conversation_id
        )
        return int((await self.session.execute(stmt)).scalar_one()) + 1

    async def context_turns_before(
        self,
        conversation_id: uuid.UUID,
        *,
        before_sequence: int,
        limit: int,
    ) -> list[tuple[ConversationMessage, ConversationMessage]]:
        """Newest eligible complete answer turns, still subject to source ACL."""
        user_message = aliased(ConversationMessage)
        assistant = aliased(ConversationMessage)
        stmt = (
            select(user_message, assistant)
            .join(
                assistant,
                and_(
                    assistant.conversation_id == user_message.conversation_id,
                    assistant.sequence == user_message.sequence + 1,
                    assistant.role == "assistant",
                ),
            )
            .where(
                user_message.conversation_id == conversation_id,
                user_message.role == "user",
                user_message.sequence < before_sequence,
                assistant.status == "complete",
                assistant.kind == "answer",
            )
            .order_by(user_message.sequence.desc())
            .limit(limit)
        )
        return [
            (user_message, assistant_message)
            for user_message, assistant_message in (await self.session.execute(stmt)).all()
        ]

    async def touch_conversation(self, conversation_id: uuid.UUID, *, updated_at: datetime) -> None:
        await self.session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(updated_at=updated_at)
        )

    async def sources_for_messages(self, message_ids: Sequence[uuid.UUID]) -> list[MessageSource]:
        ids = list(dict.fromkeys(message_ids))
        if not ids:
            return []
        stmt = (
            select(MessageSource)
            .where(MessageSource.message_id.in_(ids))
            .order_by(MessageSource.message_id, MessageSource.retrieval_rank)
        )
        return list((await self.session.execute(stmt)).scalars())
