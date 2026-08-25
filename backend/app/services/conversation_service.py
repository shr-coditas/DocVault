"""Creator-owned conversation lifecycle and immutable scope validation."""

import base64
import binascii
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from uuid6 import uuid7

from app.config import get_settings
from app.controller.conversation_controller.dto.conversation_dto import (
    ConversationCreate,
    ConversationMessageCreate,
)
from app.exceptions import NotFoundError, UnprocessableEntityError
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
from app.models.document import Document
from app.models.user import User
from app.repository.conversation_repository import ConversationRepository
from app.repository.document_repository import DocumentRepository
from app.services.ai_types import (
    Citation,
    ConversationTurn,
    QueryDecision,
    QueryExecutionContext,
    QueryOutcome,
    SearchHit,
    UnavailableDocument,
)
from app.services.audit_service import AuditService
from app.services.document_access import document_access_filter
from app.services.query_service import QueryService

DEFAULT_CONVERSATION_TITLE = "New conversation"
REDACTED_ANSWER_MESSAGE = (
    "This answer is unavailable because access to one or more of its sources changed."
)

logger = structlog.stdlib.get_logger("docvault.conversation")


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    conversation: Conversation
    documents: tuple[ConversationDocument, ...]


@dataclass(frozen=True, slots=True)
class ConversationPage:
    items: tuple[ConversationRecord, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class MessagePage:
    items: tuple["MessageRecord", ...]
    next_after_sequence: int | None


@dataclass(frozen=True, slots=True)
class MessageRecord:
    message: ConversationMessage
    sources: tuple[MessageSource, ...] = ()
    redacted: bool = False
    unavailable_documents: tuple[ConversationDocument, ...] = ()


@dataclass(frozen=True, slots=True)
class MessageSubmission:
    messages: tuple[MessageRecord, ...]


def _retrieval_kind(outcome: QueryOutcome) -> MessageKind:
    """Why a retrieving turn ends without an answer, recorded as it happened.

    Five distinct events reach here and each gets its own kind, because a
    history that flattens them is a history nobody can audit later: was the
    workspace missing the document, were the passages too thin, did the model
    write something unsafe, did it write something ungrounded, or was the
    provider simply down? Only the last is an incident.

    Order matters. A rejected draft is checked before evidence sufficiency,
    because a turn that got far enough to generate had sources the supervisor
    was content with - the failure was in the output, not in what was found.
    """
    if not outcome.hits:
        return MessageKind.NO_SOURCES
    verdict = outcome.output_verdict
    if verdict is not None and not verdict.passed:
        return MessageKind.REFUSAL if verdict.security_failure else MessageKind.ANSWER_REJECTED
    if outcome.evidence_sufficient is False:
        return MessageKind.UNSUPPORTED_EVIDENCE
    return MessageKind.GENERATION_UNAVAILABLE


@dataclass(frozen=True, slots=True)
class _FinalTurn:
    status: MessageStatus
    kind: MessageKind
    content: str
    input_tokens: int | None
    output_tokens: int | None
    hits: tuple[SearchHit, ...]
    selected_sources: tuple[SearchHit, ...]
    citations: tuple[Citation, ...]


def _source_key(hit: SearchHit) -> uuid.UUID:
    return hit.chunk_id


def _encode_cursor(updated_at: datetime, conversation_id: uuid.UUID) -> str:
    payload = json.dumps(
        {"updated_at": updated_at.isoformat(), "id": str(conversation_id)},
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()))
        if not isinstance(payload, dict):
            raise ValueError
        updated_at = datetime.fromisoformat(payload["updated_at"])
        if updated_at.tzinfo is None:
            raise ValueError
        return updated_at, uuid.UUID(payload["id"])
    except (
        AttributeError,
        binascii.Error,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
    ):
        raise UnprocessableEntityError("invalid conversation cursor") from None


class ConversationService:
    def __init__(self, session: AsyncSession, *, query: QueryService | None = None) -> None:
        self.session = session
        self.repository = ConversationRepository(session)
        self.documents = DocumentRepository(session)
        self.audit = AuditService(session)
        self.query = query

    async def create(
        self,
        actor: User,
        workspace_id: uuid.UUID,
        data: ConversationCreate,
    ) -> ConversationRecord:
        selected = await self._selected_documents(actor, workspace_id, data)
        conversation = Conversation(
            id=uuid7(),
            workspace_id=workspace_id,
            created_by=actor.id,
            title=selected[0].title if len(selected) == 1 else DEFAULT_CONVERSATION_TITLE,
            scope_mode=data.scope_mode,
        )
        self.repository.add(conversation)
        pinned = tuple(
            ConversationDocument(
                conversation_id=conversation.id,
                document_id=document.id,
                position=position,
                title_snapshot=document.title,
                file_name_snapshot=document.file_name,
            )
            for position, document in enumerate(selected)
        )
        self.repository.add_documents(pinned)
        self.audit.record(
            action="conversation.created",
            resource_type="conversation",
            resource_id=conversation.id,
            workspace_id=workspace_id,
            actor_id=actor.id,
            scope_mode=data.scope_mode.value,
            document_count=len(pinned),
        )
        await self.session.commit()
        await self.session.refresh(conversation)
        return ConversationRecord(conversation, pinned)

    async def list_owned(
        self,
        actor: User,
        workspace_id: uuid.UUID,
        *,
        limit: int,
        cursor: str | None,
    ) -> ConversationPage:
        before = _decode_cursor(cursor) if cursor else None
        rows = await self.repository.list_owned(
            workspace_id=workspace_id,
            creator_id=actor.id,
            limit=limit + 1,
            before=before,
        )
        has_more = len(rows) > limit
        rows = rows[:limit]
        documents = await self.repository.documents_for_conversations(
            [conversation.id for conversation in rows]
        )
        records = tuple(
            ConversationRecord(conversation, tuple(documents.get(conversation.id, [])))
            for conversation in rows
        )
        next_cursor = None
        if has_more and rows:
            last = rows[-1]
            next_cursor = _encode_cursor(last.updated_at, last.id)
        return ConversationPage(records, next_cursor)

    async def get_owned(
        self, actor: User, workspace_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> ConversationRecord:
        conversation = await self.repository.get_owned(
            conversation_id,
            workspace_id=workspace_id,
            creator_id=actor.id,
        )
        if conversation is None:
            raise NotFoundError("conversation not found")
        documents = await self.repository.list_documents(conversation.id)
        return ConversationRecord(conversation, tuple(documents))

    async def delete_owned(
        self, actor: User, workspace_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> None:
        conversation = await self.repository.get_owned(
            conversation_id,
            workspace_id=workspace_id,
            creator_id=actor.id,
        )
        if conversation is None:
            raise NotFoundError("conversation not found")
        self.audit.record(
            action="conversation.deleted",
            resource_type="conversation",
            resource_id=conversation.id,
            workspace_id=workspace_id,
            actor_id=actor.id,
        )
        await self.repository.delete(conversation)
        await self.session.commit()

    async def list_messages(
        self,
        actor: User,
        workspace_id: uuid.UUID,
        conversation_id: uuid.UUID,
        *,
        after_sequence: int,
        limit: int,
    ) -> MessagePage:
        actor_id = actor.id
        await self.get_owned(actor, workspace_id, conversation_id)
        rows = await self.repository.list_messages_after(
            conversation_id,
            after_sequence=after_sequence,
            limit=limit + 1,
        )
        has_more = len(rows) > limit
        rows = rows[:limit]
        return MessagePage(
            await self._project_messages(actor_id, workspace_id, rows),
            rows[-1].sequence if has_more and rows else None,
        )

    async def submit_message(
        self,
        actor: User,
        workspace_id: uuid.UUID,
        conversation_id: uuid.UUID,
        data: ConversationMessageCreate,
    ) -> MessageSubmission:
        """Generate synchronously, then persist one completed message pair."""
        if self.query is None:
            raise RuntimeError("conversation query service is not configured")

        actor_id = actor.id
        conversation = await self.repository.get_owned(
            conversation_id,
            workspace_id=workspace_id,
            creator_id=actor_id,
        )
        if conversation is None:
            await self.session.rollback()
            raise NotFoundError("conversation not found")
        before_sequence = await self.repository.next_sequence(conversation_id)

        try:

            async def load_context() -> QueryExecutionContext:
                return await self._query_context(
                    actor_id,
                    workspace_id,
                    conversation_id,
                    before_sequence=before_sequence,
                )

            outcome = await self.query.handle(
                actor,
                workspace_id,
                data.content,
                context_loader=load_context,
            )
            final = self._final_from_outcome(outcome)
            # Search uses transaction-local database settings. Close the read
            # transaction before the short message-write transaction.
            await self.session.rollback()
        except Exception as exc:
            await self.session.rollback()
            logger.error(
                "conversation_turn_execution_failed",
                conversation_id=str(conversation_id),
                actor_id=str(actor_id),
                failure_type=type(exc).__name__,
                message_chars=len(data.content),
            )
            raise

        conversation = await self.repository.get_owned_for_update(
            conversation_id,
            workspace_id=workspace_id,
            creator_id=actor_id,
        )
        if conversation is None:
            await self.session.rollback()
            raise NotFoundError("conversation not found")

        now = datetime.now(UTC)
        first_sequence = await self.repository.next_sequence(conversation_id)
        user_message = ConversationMessage(
            id=uuid7(),
            conversation_id=conversation_id,
            sequence=first_sequence,
            role=MessageRole.USER,
            status=MessageStatus.COMPLETE,
            kind=None,
            content=data.content,
            input_tokens=None,
            output_tokens=None,
        )
        assistant_message = ConversationMessage(
            id=uuid7(),
            conversation_id=conversation_id,
            sequence=first_sequence + 1,
            role=MessageRole.ASSISTANT,
            status=final.status,
            kind=final.kind,
            content=final.content,
            input_tokens=final.input_tokens,
            output_tokens=final.output_tokens,
        )
        self.repository.add_messages([user_message, assistant_message])
        self.repository.add_sources(self._message_sources(assistant_message.id, final))
        await self.repository.touch_conversation(conversation_id, updated_at=now)
        await self.session.commit()
        await self.session.refresh(user_message)
        await self.session.refresh(assistant_message)
        logger.info(
            "conversation_exchange_saved",
            conversation_id=str(conversation_id),
            actor_id=str(actor_id),
            kind=final.kind.value,
            hits=len(final.hits),
            supplied_sources=len(final.selected_sources),
            input_tokens=final.input_tokens,
            output_tokens=final.output_tokens,
        )
        return MessageSubmission(
            await self._project_messages(
                actor_id,
                workspace_id,
                [user_message, assistant_message],
            )
        )

    @staticmethod
    def _final_from_outcome(outcome: QueryOutcome) -> _FinalTurn:
        if outcome.answer is not None:
            answer = outcome.answer
            return _FinalTurn(
                status=MessageStatus.COMPLETE,
                kind=MessageKind.ANSWER,
                content=answer.text,
                input_tokens=answer.input_tokens,
                output_tokens=answer.output_tokens,
                hits=outcome.hits,
                selected_sources=outcome.selected_sources,
                citations=answer.citations,
            )

        if outcome.decision is QueryDecision.RETRIEVE:
            kind = _retrieval_kind(outcome)
        elif outcome.decision is QueryDecision.BLOCK:
            kind = MessageKind.REFUSAL
        elif outcome.decision is QueryDecision.DECLINE:
            kind = MessageKind.DECLINE
        elif outcome.decision is QueryDecision.CLARIFY:
            kind = MessageKind.CLARIFICATION
        elif outcome.decision is QueryDecision.SCOPE_UNAVAILABLE:
            kind = MessageKind.SCOPE_UNAVAILABLE
        else:
            kind = MessageKind.CHITCHAT
        assert outcome.message is not None
        return _FinalTurn(
            status=MessageStatus.COMPLETE,
            kind=kind,
            content=outcome.message,
            input_tokens=None,
            output_tokens=None,
            hits=outcome.hits,
            selected_sources=outcome.selected_sources,
            citations=(),
        )

    @staticmethod
    def _message_sources(assistant_message_id: uuid.UUID, final: _FinalTurn) -> list[MessageSource]:
        selected = {_source_key(hit) for hit in final.selected_sources}
        citations = {citation.chunk_id: citation.marker for citation in final.citations}
        return [
            MessageSource(
                id=uuid7(),
                message_id=assistant_message_id,
                document_id=hit.document_id,
                chunk_id=hit.chunk_id,
                document_title_snapshot=hit.document_title,
                section_path=hit.section_path,
                chunk_type=hit.chunk_type,
                retrieval_rank=rank,
                supplied_to_model=_source_key(hit) in selected,
                citation_marker=citations.get(_source_key(hit)),
            )
            for rank, hit in enumerate(final.hits, start=1)
        ]

    async def _query_context(
        self,
        actor_id: uuid.UUID,
        workspace_id: uuid.UUID,
        conversation_id: uuid.UUID,
        *,
        before_sequence: int,
    ) -> QueryExecutionContext:
        """Resolve current selected scope, then load only access-safe history."""
        pinned = await self.repository.list_documents(conversation_id)
        access = await document_access_filter(self.session, actor_id, workspace_id)
        document_ids: tuple[uuid.UUID, ...] | None = None
        unavailable: tuple[UnavailableDocument, ...] = ()
        if pinned:
            accessible = await self.documents.accessible_active_by_ids(
                workspace_id,
                [document.document_id for document in pinned],
                access=access,
            )
            accessible_ids = {document.id for document in accessible}
            document_ids = tuple(
                document.document_id
                for document in pinned
                if document.document_id in accessible_ids
            )
            unavailable = tuple(
                UnavailableDocument(
                    document.document_id,
                    document.title_snapshot,
                    document.file_name_snapshot,
                )
                for document in pinned
                if document.document_id not in accessible_ids
            )
            if not document_ids:
                return QueryExecutionContext(
                    document_ids=(),
                    unavailable_documents=unavailable,
                )

        history = await self._context_history(
            conversation_id,
            workspace_id,
            before_sequence=before_sequence,
            access=access,
        )
        return QueryExecutionContext(
            document_ids=document_ids,
            history=history,
            unavailable_documents=unavailable,
        )

    async def _context_history(
        self,
        conversation_id: uuid.UUID,
        workspace_id: uuid.UUID,
        *,
        before_sequence: int,
        access: tuple[uuid.UUID, list[uuid.UUID]] | None,
    ) -> tuple[ConversationTurn, ...]:
        settings = get_settings()
        maximum = settings.resolver_history_max_turns
        if maximum <= 0:
            return ()
        candidates = await self.repository.context_turns_before(
            conversation_id,
            before_sequence=before_sequence,
            limit=maximum * 4,
        )
        if not candidates:
            return ()

        assistant_ids = [assistant.id for _, assistant in candidates]
        sources = await self.repository.sources_for_messages(assistant_ids)
        accessible = await self.documents.accessible_active_by_ids(
            workspace_id,
            [source.document_id for source in sources],
            access=access,
        )
        accessible_ids = {document.id for document in accessible}
        by_message: dict[uuid.UUID, list[MessageSource]] = {}
        for source in sources:
            by_message.setdefault(source.message_id, []).append(source)

        selected: list[ConversationTurn] = []
        for user_message, assistant in candidates:
            supplied = [
                source for source in by_message.get(assistant.id, []) if source.supplied_to_model
            ]
            if not supplied or any(source.document_id not in accessible_ids for source in supplied):
                continue
            user_content = user_message.content or ""
            assistant_content = assistant.content or ""
            if not user_content or not assistant_content:
                continue
            selected.append(ConversationTurn(user_content, assistant_content))
            if len(selected) >= maximum:
                break
        selected.reverse()
        return tuple(selected)

    async def _project_messages(
        self,
        actor_id: uuid.UUID,
        workspace_id: uuid.UUID,
        messages: list[ConversationMessage],
    ) -> tuple[MessageRecord, ...]:
        if not messages:
            return ()
        sources = await self.repository.sources_for_messages([message.id for message in messages])
        pinned = await self.repository.list_documents(messages[0].conversation_id)
        access = await document_access_filter(self.session, actor_id, workspace_id)
        accessible = await self.documents.accessible_active_by_ids(
            workspace_id,
            [
                *[source.document_id for source in sources],
                *[document.document_id for document in pinned],
            ],
            access=access,
        )
        accessible_ids = {document.id for document in accessible}
        unavailable = tuple(
            document for document in pinned if document.document_id not in accessible_ids
        )
        grouped: dict[uuid.UUID, list[MessageSource]] = {}
        for source in sources:
            grouped.setdefault(source.message_id, []).append(source)

        records: list[MessageRecord] = []
        for message in messages:
            message_sources = grouped.get(message.id, [])
            redacted = message.kind == MessageKind.ANSWER and any(
                source.supplied_to_model and source.document_id not in accessible_ids
                for source in message_sources
            )
            projected_sources: tuple[MessageSource, ...] = ()
            if not redacted:
                projected_sources = tuple(
                    source for source in message_sources if source.document_id in accessible_ids
                )
            records.append(
                MessageRecord(
                    message,
                    projected_sources,
                    redacted,
                    unavailable if message.role == MessageRole.ASSISTANT else (),
                )
            )
        return tuple(records)

    async def _selected_documents(
        self,
        actor: User,
        workspace_id: uuid.UUID,
        data: ConversationCreate,
    ) -> list[Document]:
        if data.scope_mode is ConversationScope.WORKSPACE:
            return []
        requested = data.document_ids or []
        access = await document_access_filter(self.session, actor.id, workspace_id)
        documents = await self.documents.accessible_active_by_ids(
            workspace_id,
            requested,
            access=access,
        )
        by_id = {document.id: document for document in documents}
        if len(by_id) != len(requested):
            raise NotFoundError("one or more documents not found")
        return [by_id[document_id] for document_id in requested]
