import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from app.controller.conversation_controller import conversation_controller
from app.controller.conversation_controller.dto.conversation_dto import (
    ConversationCreate,
    ConversationMessageCreate,
    ConversationMessagePageOut,
    ConversationMessageSubmissionOut,
    ConversationOut,
    ConversationPageOut,
)
from app.dependencies import (
    DbSession,
    QueryServiceDep,
    require_permission,
)
from app.models.user import User
from app.services.conversation_service import ConversationService
from app.utils.rbac_catalog import Perm

router = APIRouter(
    prefix="/workspaces/{workspace_id}/conversations",
    tags=["conversations"],
)


def get_conversation_service(db: DbSession) -> ConversationService:
    return ConversationService(db)


def get_conversation_submission_service(
    db: DbSession,
    query: QueryServiceDep,
) -> ConversationService:
    return ConversationService(db, query=query)


ServiceDep = Annotated[ConversationService, Depends(get_conversation_service)]
SubmissionServiceDep = Annotated[
    ConversationService,
    Depends(get_conversation_submission_service),
]
CanReadDocuments = Annotated[User, Depends(require_permission(Perm.DOCUMENT_READ))]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_conversation(
    workspace_id: uuid.UUID,
    data: ConversationCreate,
    user: CanReadDocuments,
    service: ServiceDep,
) -> ConversationOut:
    """Create a creator-private conversation with an immutable search scope."""
    return await conversation_controller.create_conversation(workspace_id, data, user, service)


@router.get("")
async def list_conversations(
    workspace_id: uuid.UUID,
    user: CanReadDocuments,
    service: ServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query(max_length=512)] = None,
) -> ConversationPageOut:
    return await conversation_controller.list_conversations(
        workspace_id,
        user,
        service,
        limit=limit,
        cursor=cursor,
    )


@router.get("/{conversation_id}")
async def get_conversation(
    workspace_id: uuid.UUID,
    conversation_id: uuid.UUID,
    user: CanReadDocuments,
    service: ServiceDep,
) -> ConversationOut:
    return await conversation_controller.get_conversation(
        workspace_id, conversation_id, user, service
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    workspace_id: uuid.UUID,
    conversation_id: uuid.UUID,
    user: CanReadDocuments,
    service: ServiceDep,
) -> None:
    await conversation_controller.delete_conversation(workspace_id, conversation_id, user, service)


@router.get("/{conversation_id}/messages")
async def list_messages(
    workspace_id: uuid.UUID,
    conversation_id: uuid.UUID,
    user: CanReadDocuments,
    service: ServiceDep,
    after_sequence: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> ConversationMessagePageOut:
    return await conversation_controller.list_messages(
        workspace_id,
        conversation_id,
        user,
        service,
        after_sequence=after_sequence,
        limit=limit,
    )


@router.post(
    "/{conversation_id}/messages",
    response_model=ConversationMessageSubmissionOut,
    status_code=status.HTTP_201_CREATED,
)
async def submit_message(
    workspace_id: uuid.UUID,
    conversation_id: uuid.UUID,
    data: ConversationMessageCreate,
    user: CanReadDocuments,
    service: SubmissionServiceDep,
) -> ConversationMessageSubmissionOut:
    """Generate synchronously, then persist and return the completed exchange."""
    return await conversation_controller.submit_message(
        workspace_id,
        conversation_id,
        data,
        user,
        service,
    )
