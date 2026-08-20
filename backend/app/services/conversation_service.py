"""Creator-owned conversation lifecycle and immutable scope validation."""

import base64
import binascii
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from uuid6 import uuid7

from app.config import get_settings
from app.controller.conversation_controller.dto.conversation_dto import (
    ConversationCreate,
    ConversationMessageCreate,
)
from app.exceptions import ConflictError, NotFoundError, UnprocessableEntityError
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
from app.repository.conversation_repository import ConversationRepository, ResolvedSource
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
from app.services.token_counting import ConservativeGenerationTokenCounter

DEFAULT_CONVERSATION_TITLE = "New conversation"
TURN_LEASE_DURATION = timedelta(minutes=2)
TURN_RETRY_AFTER_SECONDS = 2
TURN_ERROR_MESSAGE = "The answer could not be completed. Please try again."
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
class SourceRecord:
    source: MessageSource
    resolved_chunk_id: uuid.UUID | None
    relocated: bool


@dataclass(frozen=True, slots=True)
class MessageRecord:
    message: ConversationMessage
    sources: tuple[SourceRecord, ...] = ()
    redacted: bool = False
    unavailable_documents: tuple[ConversationDocument, ...] = ()


@dataclass(frozen=True, slots=True)
class TurnRecord:
    turn_id: uuid.UUID
    messages: tuple[MessageRecord, ...]

    @property
    def assistant(self) -> ConversationMessage:
        return next(
            record.message
            for record in self.messages
            if record.message.role == MessageRole.ASSISTANT
        )


@dataclass(frozen=True, slots=True)
class TurnSubmission:
    turn: TurnRecord
    created: bool


@dataclass(frozen=True, slots=True)
class _TurnClaim:
    turn_id: uuid.UUID
    user_message_id: uuid.UUID
    user_sequence: int
    assistant_message_id: uuid.UUID
    lease_token: uuid.UUID | None
    content: str
    execute: bool
    created: bool


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
    context_eligible: bool
    resolved_query: str | None
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    hits: tuple[SearchHit, ...]
    selected_sources: tuple[SearchHit, ...]
    citations: tuple[Citation, ...]


def _source_key(hit: SearchHit) -> tuple[uuid.UUID, int, str]:
    return hit.document_id, hit.index_generation, hit.logical_key


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
    ) -> TurnSubmission:
        """Persist a leased turn, execute outside the lock, then finalize atomically."""
        if self.query is None:
            raise RuntimeError("conversation query service is not configured")

        actor_id = actor.id
        claim = await self._claim_turn(actor_id, workspace_id, conversation_id, data)
        if not claim.execute:
            return TurnSubmission(
                await self._load_turn(actor_id, workspace_id, conversation_id, claim.turn_id),
                created=False,
            )

        assert claim.lease_token is not None
        try:

            async def load_context() -> QueryExecutionContext:
                return await self._query_context(
                    actor_id,
                    workspace_id,
                    conversation_id,
                    before_sequence=claim.user_sequence,
                )

            outcome = await self.query.handle(
                actor,
                workspace_id,
                claim.content,
                context_loader=load_context,
            )
            final = self._final_from_outcome(outcome)
            # Search uses transaction-local database settings. Close that read
            # transaction before the independently fenced write transaction.
            await self.session.rollback()
        except Exception as exc:
            # Never log model/search exception text: provider messages can carry
            # request details. The type and identifiers are enough operationally.
            await self.session.rollback()
            logger.error(
                "conversation_turn_execution_failed",
                conversation_id=str(conversation_id),
                turn_id=str(claim.turn_id),
                actor_id=str(actor_id),
                failure_type=type(exc).__name__,
                message_chars=len(claim.content),
            )
            final = _FinalTurn(
                status=MessageStatus.FAILED,
                kind=MessageKind.ERROR,
                content=TURN_ERROR_MESSAGE,
                context_eligible=False,
                resolved_query=None,
                model=None,
                input_tokens=None,
                output_tokens=None,
                hits=(),
                selected_sources=(),
                citations=(),
            )

        won = await self._finalize_turn(claim, conversation_id, final)
        if not won:
            logger.info(
                "conversation_turn_fence_lost",
                conversation_id=str(conversation_id),
                turn_id=str(claim.turn_id),
                actor_id=str(actor_id),
            )
        turn = await self._load_turn(actor_id, workspace_id, conversation_id, claim.turn_id)
        return TurnSubmission(turn, created=claim.created)

    async def get_turn(
        self,
        actor: User,
        workspace_id: uuid.UUID,
        conversation_id: uuid.UUID,
        turn_id: uuid.UUID,
    ) -> TurnRecord:
        await self.get_owned(actor, workspace_id, conversation_id)
        return await self._load_turn(actor.id, workspace_id, conversation_id, turn_id)

    async def _claim_turn(
        self,
        actor_id: uuid.UUID,
        workspace_id: uuid.UUID,
        conversation_id: uuid.UUID,
        data: ConversationMessageCreate,
    ) -> _TurnClaim:
        """Transaction A: serialize submissions and commit before model work."""
        conversation = await self.repository.get_owned_for_update(
            conversation_id,
            workspace_id=workspace_id,
            creator_id=actor_id,
        )
        if conversation is None:
            await self.session.rollback()
            raise NotFoundError("conversation not found")

        existing = await self.repository.turn_for_client_message(
            conversation_id, data.client_message_id
        )
        if existing:
            user_message, assistant = self._turn_pair(existing)
            if assistant.status != MessageStatus.PENDING:
                await self.session.commit()
                return _TurnClaim(
                    assistant.turn_id,
                    user_message.id,
                    user_message.sequence,
                    assistant.id,
                    None,
                    user_message.content or "",
                    execute=False,
                    created=False,
                )

            now = datetime.now(UTC)
            assert assistant.lease_expires_at is not None
            if assistant.lease_expires_at > now:
                await self.session.commit()
                return _TurnClaim(
                    assistant.turn_id,
                    user_message.id,
                    user_message.sequence,
                    assistant.id,
                    assistant.lease_token,
                    user_message.content or "",
                    execute=False,
                    created=False,
                )

            lease_token = uuid7()
            assistant.lease_token = lease_token
            assistant.lease_expires_at = now + TURN_LEASE_DURATION
            assistant.updated_at = now
            await self.session.commit()
            logger.info(
                "conversation_turn_lease_recovered",
                conversation_id=str(conversation_id),
                turn_id=str(assistant.turn_id),
                actor_id=str(actor_id),
            )
            return _TurnClaim(
                assistant.turn_id,
                user_message.id,
                user_message.sequence,
                assistant.id,
                lease_token,
                user_message.content or "",
                execute=True,
                created=False,
            )

        if await self.repository.pending_assistant(conversation_id) is not None:
            await self.session.rollback()
            raise ConflictError("another conversation turn is pending")

        now = datetime.now(UTC)
        turn_id = uuid7()
        user_message_id = uuid7()
        assistant_message_id = uuid7()
        lease_token = uuid7()
        first_sequence = await self.repository.next_sequence(conversation_id)
        self.repository.add_messages(
            [
                ConversationMessage(
                    id=user_message_id,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    sequence=first_sequence,
                    role=MessageRole.USER,
                    status=MessageStatus.COMPLETE,
                    kind=None,
                    content=data.content,
                    context_eligible=False,
                    resolved_query=None,
                    client_message_id=data.client_message_id,
                    lease_token=None,
                    lease_expires_at=None,
                ),
                ConversationMessage(
                    id=assistant_message_id,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    sequence=first_sequence + 1,
                    role=MessageRole.ASSISTANT,
                    status=MessageStatus.PENDING,
                    kind=None,
                    content=None,
                    context_eligible=False,
                    resolved_query=None,
                    client_message_id=None,
                    lease_token=lease_token,
                    lease_expires_at=now + TURN_LEASE_DURATION,
                ),
            ]
        )
        await self.repository.touch_conversation(conversation_id, updated_at=now)
        await self.session.commit()
        logger.info(
            "conversation_turn_created",
            conversation_id=str(conversation_id),
            turn_id=str(turn_id),
            actor_id=str(actor_id),
            message_chars=len(data.content),
        )
        return _TurnClaim(
            turn_id,
            user_message_id,
            first_sequence,
            assistant_message_id,
            lease_token,
            data.content,
            execute=True,
            created=True,
        )

    @staticmethod
    def _turn_pair(
        messages: list[ConversationMessage],
    ) -> tuple[ConversationMessage, ConversationMessage]:
        user_message = next(message for message in messages if message.role == MessageRole.USER)
        assistant = next(message for message in messages if message.role == MessageRole.ASSISTANT)
        return user_message, assistant

    @staticmethod
    def _final_from_outcome(outcome: QueryOutcome) -> _FinalTurn:
        if outcome.answer is not None:
            answer = outcome.answer
            return _FinalTurn(
                status=MessageStatus.COMPLETE,
                kind=MessageKind.ANSWER,
                content=answer.text,
                context_eligible=True,
                resolved_query=outcome.query,
                model=answer.model,
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
            context_eligible=False,
            resolved_query=(outcome.query if outcome.retrieval_performed else None),
            model=None,
            input_tokens=None,
            output_tokens=None,
            hits=outcome.hits,
            selected_sources=outcome.selected_sources,
            citations=(),
        )

    async def _finalize_turn(
        self,
        claim: _TurnClaim,
        conversation_id: uuid.UUID,
        final: _FinalTurn,
    ) -> bool:
        """Transaction B: fence the writer, then commit message and ledger together."""
        assert claim.lease_token is not None
        now = datetime.now(UTC)
        won = await self.repository.finalize_assistant(
            claim.assistant_message_id,
            lease_token=claim.lease_token,
            status=final.status.value,
            kind=final.kind.value,
            content=final.content,
            model=final.model,
            input_tokens=final.input_tokens,
            output_tokens=final.output_tokens,
            updated_at=now,
        )
        if not won:
            await self.session.rollback()
            return False

        await self.repository.update_user_context(
            claim.user_message_id,
            context_eligible=final.context_eligible,
            resolved_query=final.resolved_query,
            updated_at=now,
        )
        self.repository.add_sources(self._message_sources(claim.assistant_message_id, final))
        await self.repository.touch_conversation(conversation_id, updated_at=now)
        await self.session.commit()
        logger.info(
            "conversation_turn_finalized",
            conversation_id=str(conversation_id),
            turn_id=str(claim.turn_id),
            status=final.status.value,
            kind=final.kind.value,
            hits=len(final.hits),
            supplied_sources=len(final.selected_sources),
            model=final.model,
            input_tokens=final.input_tokens,
            output_tokens=final.output_tokens,
        )
        return True

    @staticmethod
    def _message_sources(assistant_message_id: uuid.UUID, final: _FinalTurn) -> list[MessageSource]:
        selected = {_source_key(hit) for hit in final.selected_sources}
        citations = {
            (
                citation.document_id,
                citation.index_generation,
                citation.logical_key,
            ): citation.marker
            for citation in final.citations
        }
        return [
            MessageSource(
                id=uuid7(),
                message_id=assistant_message_id,
                document_id=hit.document_id,
                chunk_id=hit.chunk_id,
                index_generation=hit.index_generation,
                logical_key=hit.logical_key,
                document_title_snapshot=hit.document_title,
                heading=hit.heading,
                breadcrumb=hit.breadcrumb,
                page_numbers=list(hit.page_numbers),
                source_spans=[asdict(span) for span in hit.source_spans],
                retrieval_rank=rank,
                supplied_to_model=_source_key(hit) in selected,
                citation_marker=citations.get(_source_key(hit)),
            )
            for rank, hit in enumerate(final.hits, start=1)
        ]

    async def _load_turn(
        self,
        actor_id: uuid.UUID,
        workspace_id: uuid.UUID,
        conversation_id: uuid.UUID,
        turn_id: uuid.UUID,
    ) -> TurnRecord:
        messages = await self.repository.messages_for_turn(conversation_id, turn_id)
        if len(messages) != 2:
            raise NotFoundError("conversation turn not found")
        return TurnRecord(
            turn_id,
            await self._project_messages(actor_id, workspace_id, messages),
        )

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
        if maximum <= 0 or settings.resolver_history_token_budget <= 0:
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

        counter = ConservativeGenerationTokenCounter()
        selected: list[ConversationTurn] = []
        spent = 0
        for user_message, assistant in candidates:
            supplied = [
                source for source in by_message.get(assistant.id, []) if source.supplied_to_model
            ]
            if not supplied or any(source.document_id not in accessible_ids for source in supplied):
                continue
            user_content = user_message.content or ""
            assistant_content = assistant.content or ""
            cost = counter.count_tokens(user_content) + counter.count_tokens(assistant_content)
            if cost <= 0 or spent + cost > settings.resolver_history_token_budget:
                continue
            selected.append(ConversationTurn(user_content, assistant_content))
            spent += cost
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
        visible_sources = [source for source in sources if source.document_id in accessible_ids]
        resolved = await self.repository.resolve_source_chunks(
            [source.id for source in visible_sources],
            workspace_id=workspace_id,
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
            projected_sources: tuple[SourceRecord, ...] = ()
            if not redacted:
                projected_sources = tuple(
                    self._source_record(source, resolved.get(source.id))
                    for source in message_sources
                    if source.document_id in accessible_ids
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

    @staticmethod
    def _source_record(source: MessageSource, resolved: ResolvedSource | None) -> SourceRecord:
        chunk = resolved.chunk if resolved is not None else None
        return SourceRecord(
            source,
            resolved_chunk_id=chunk.id if chunk is not None else None,
            relocated=resolved.relocated if resolved is not None else False,
        )

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
