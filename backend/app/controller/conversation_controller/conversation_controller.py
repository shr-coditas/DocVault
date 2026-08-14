import uuid

from app.controller.conversation_controller.dto.conversation_dto import (
    ConversationCreate,
    ConversationDocumentOut,
    ConversationMessageCreate,
    ConversationMessageOut,
    ConversationMessagePageOut,
    ConversationMessageSourceOut,
    ConversationOut,
    ConversationPageOut,
    ConversationTurnOut,
    UnavailableConversationDocumentOut,
)
from app.models.user import User
from app.services.conversation_service import (
    REDACTED_ANSWER_MESSAGE,
    TURN_RETRY_AFTER_SECONDS,
    ConversationRecord,
    ConversationService,
    MessageRecord,
    TurnRecord,
)


def _conversation_out(record: ConversationRecord) -> ConversationOut:
    conversation = record.conversation
    return ConversationOut(
        id=conversation.id,
        workspace_id=conversation.workspace_id,
        title=conversation.title,
        scope_mode=conversation.scope_mode,
        documents=[
            ConversationDocumentOut(
                document_id=document.document_id,
                position=document.position,
                title=document.title_snapshot,
                file_name=document.file_name_snapshot,
            )
            for document in record.documents
        ],
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def _message_out(record: MessageRecord) -> ConversationMessageOut:
    message = record.message
    unavailable_documents = [
        UnavailableConversationDocumentOut(
            document_id=document.document_id,
            title=document.title_snapshot,
            file_name=document.file_name_snapshot,
        )
        for document in record.unavailable_documents
    ]
    if record.redacted:
        return ConversationMessageOut(
            id=message.id,
            turn_id=message.turn_id,
            sequence=message.sequence,
            role=message.role,
            status=message.status,
            kind="redacted",
            content=REDACTED_ANSWER_MESSAGE,
            context_eligible=False,
            redacted=True,
            sources=[],
            scope_degraded=bool(unavailable_documents),
            unavailable_documents=unavailable_documents,
            client_message_id=message.client_message_id,
            model=None,
            input_tokens=None,
            output_tokens=None,
            created_at=message.created_at,
            updated_at=message.updated_at,
        )
    return ConversationMessageOut(
        id=message.id,
        turn_id=message.turn_id,
        sequence=message.sequence,
        role=message.role,
        status=message.status,
        kind=message.kind,
        content=message.content,
        context_eligible=message.context_eligible,
        redacted=False,
        sources=[
            ConversationMessageSourceOut(
                id=source_record.source.id,
                document_id=source_record.source.document_id,
                document_title=source_record.source.document_title_snapshot,
                chunk_id=source_record.source.chunk_id,
                resolved_chunk_id=source_record.resolved_chunk_id,
                relocated=source_record.relocated,
                index_generation=source_record.source.index_generation,
                logical_key=source_record.source.logical_key,
                heading=source_record.source.heading,
                breadcrumb=source_record.source.breadcrumb,
                page_numbers=source_record.source.page_numbers,
                source_spans=source_record.source.source_spans,
                retrieval_rank=source_record.source.retrieval_rank,
                supplied_to_model=source_record.source.supplied_to_model,
                citation_marker=source_record.source.citation_marker,
            )
            for source_record in record.sources
        ],
        scope_degraded=bool(unavailable_documents),
        unavailable_documents=unavailable_documents,
        client_message_id=message.client_message_id,
        model=message.model,
        input_tokens=message.input_tokens,
        output_tokens=message.output_tokens,
        created_at=message.created_at,
        updated_at=message.updated_at,
    )


def _turn_out(
    workspace_id: uuid.UUID,
    conversation_id: uuid.UUID,
    turn: TurnRecord,
) -> ConversationTurnOut:
    assistant_record = next(
        record for record in turn.messages if record.message.role == "assistant"
    )
    status = assistant_record.message.status
    unavailable_documents = [
        UnavailableConversationDocumentOut(
            document_id=document.document_id,
            title=document.title_snapshot,
            file_name=document.file_name_snapshot,
        )
        for document in assistant_record.unavailable_documents
    ]
    return ConversationTurnOut(
        turn_id=turn.turn_id,
        status=status,
        messages=[_message_out(message) for message in turn.messages],
        status_url=(
            f"/api/v1/workspaces/{workspace_id}/conversations/"
            f"{conversation_id}/turns/{turn.turn_id}"
        ),
        retry_after_seconds=(TURN_RETRY_AFTER_SECONDS if status == "pending" else None),
        scope_degraded=bool(unavailable_documents),
        unavailable_documents=unavailable_documents,
    )


async def create_conversation(
    workspace_id: uuid.UUID,
    data: ConversationCreate,
    user: User,
    service: ConversationService,
) -> ConversationOut:
    return _conversation_out(await service.create(user, workspace_id, data))


async def list_conversations(
    workspace_id: uuid.UUID,
    user: User,
    service: ConversationService,
    *,
    limit: int,
    cursor: str | None,
) -> ConversationPageOut:
    page = await service.list_owned(user, workspace_id, limit=limit, cursor=cursor)
    return ConversationPageOut(
        items=[_conversation_out(record) for record in page.items],
        next_cursor=page.next_cursor,
    )


async def get_conversation(
    workspace_id: uuid.UUID,
    conversation_id: uuid.UUID,
    user: User,
    service: ConversationService,
) -> ConversationOut:
    return _conversation_out(await service.get_owned(user, workspace_id, conversation_id))


async def delete_conversation(
    workspace_id: uuid.UUID,
    conversation_id: uuid.UUID,
    user: User,
    service: ConversationService,
) -> None:
    await service.delete_owned(user, workspace_id, conversation_id)


async def list_messages(
    workspace_id: uuid.UUID,
    conversation_id: uuid.UUID,
    user: User,
    service: ConversationService,
    *,
    after_sequence: int,
    limit: int,
) -> ConversationMessagePageOut:
    page = await service.list_messages(
        user,
        workspace_id,
        conversation_id,
        after_sequence=after_sequence,
        limit=limit,
    )
    return ConversationMessagePageOut(
        items=[_message_out(message) for message in page.items],
        next_after_sequence=page.next_after_sequence,
    )


async def submit_message(
    workspace_id: uuid.UUID,
    conversation_id: uuid.UUID,
    data: ConversationMessageCreate,
    user: User,
    service: ConversationService,
) -> tuple[ConversationTurnOut, bool]:
    submission = await service.submit_message(user, workspace_id, conversation_id, data)
    return (
        _turn_out(workspace_id, conversation_id, submission.turn),
        submission.created,
    )


async def get_turn(
    workspace_id: uuid.UUID,
    conversation_id: uuid.UUID,
    turn_id: uuid.UUID,
    user: User,
    service: ConversationService,
) -> ConversationTurnOut:
    turn = await service.get_turn(user, workspace_id, conversation_id, turn_id)
    return _turn_out(workspace_id, conversation_id, turn)
