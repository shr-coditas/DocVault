import uuid

from app.controller.conversation_controller.dto.conversation_dto import (
    ConversationCreate,
    ConversationDocumentOut,
    ConversationMessageCreate,
    ConversationMessageOut,
    ConversationMessagePageOut,
    ConversationMessageSourceOut,
    ConversationMessageSubmissionOut,
    ConversationOut,
    ConversationPageOut,
    UnavailableConversationDocumentOut,
)
from app.models.user import User
from app.services.conversation_service import (
    REDACTED_ANSWER_MESSAGE,
    ConversationRecord,
    ConversationService,
    MessageRecord,
    MessageSubmission,
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
            sequence=message.sequence,
            role=message.role,
            status=message.status,
            kind="redacted",
            content=REDACTED_ANSWER_MESSAGE,
            redacted=True,
            sources=[],
            unavailable_documents=unavailable_documents,
            input_tokens=None,
            output_tokens=None,
            created_at=message.created_at,
            updated_at=message.updated_at,
        )
    return ConversationMessageOut(
        id=message.id,
        sequence=message.sequence,
        role=message.role,
        status=message.status,
        kind=message.kind,
        content=message.content,
        redacted=False,
        sources=[
            ConversationMessageSourceOut(
                id=source.id,
                document_id=source.document_id,
                document_title=source.document_title_snapshot,
                chunk_id=source.chunk_id,
                section_path=source.section_path,
                chunk_type=source.chunk_type,
                retrieval_rank=source.retrieval_rank,
                supplied_to_model=source.supplied_to_model,
                citation_marker=source.citation_marker,
            )
            for source in record.sources
        ],
        unavailable_documents=unavailable_documents,
        input_tokens=message.input_tokens,
        output_tokens=message.output_tokens,
        created_at=message.created_at,
        updated_at=message.updated_at,
    )


def _submission_out(submission: MessageSubmission) -> ConversationMessageSubmissionOut:
    return ConversationMessageSubmissionOut(
        messages=[_message_out(message) for message in submission.messages],
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
) -> ConversationMessageSubmissionOut:
    submission = await service.submit_message(user, workspace_id, conversation_id, data)
    return _submission_out(submission)
