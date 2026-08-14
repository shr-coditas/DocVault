import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response, status

from app.controller.conversation_controller import conversation_controller
from app.controller.conversation_controller.dto.conversation_dto import (
    ConversationCreate,
    ConversationMessageCreate,
    ConversationMessagePageOut,
    ConversationOut,
    ConversationPageOut,
    ConversationTurnOut,
)
from app.dependencies import (
    ChatModelDep,
    ContextualResolverDep,
    DbSession,
    EmbedderDep,
    RerankerDep,
    require_permission,
)
from app.models.user import User
from app.services.answer_service import AnswerService
from app.services.conversation_service import ConversationService
from app.services.query_service import QueryService
from app.services.search_service import SearchService
from app.utils.rbac_catalog import Perm

router = APIRouter(
    prefix="/workspaces/{workspace_id}/conversations",
    tags=["conversations"],
)


def get_conversation_service(db: DbSession) -> ConversationService:
    return ConversationService(db)


def get_conversation_turn_service(
    db: DbSession,
    embedder: EmbedderDep,
    reranker: RerankerDep,
    model: ChatModelDep,
    resolver: ContextualResolverDep,
) -> ConversationService:
    query = QueryService(
        SearchService(db, embedder, reranker=reranker),
        answers=AnswerService(model),
        resolver=resolver,
    )
    return ConversationService(db, query=query)


ServiceDep = Annotated[ConversationService, Depends(get_conversation_service)]
TurnServiceDep = Annotated[ConversationService, Depends(get_conversation_turn_service)]
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


@router.post("/{conversation_id}/messages", response_model=ConversationTurnOut)
async def submit_message(
    workspace_id: uuid.UUID,
    conversation_id: uuid.UUID,
    data: ConversationMessageCreate,
    response: Response,
    user: CanReadDocuments,
    service: TurnServiceDep,
) -> ConversationTurnOut:
    """Persist before generation; return or resume the idempotent leased turn."""
    turn, created = await conversation_controller.submit_message(
        workspace_id,
        conversation_id,
        data,
        user,
        service,
    )
    if turn.status == "pending":
        response.status_code = status.HTTP_202_ACCEPTED
    elif created:
        response.status_code = status.HTTP_201_CREATED
    return turn


@router.get("/{conversation_id}/turns/{turn_id}")
async def get_turn(
    workspace_id: uuid.UUID,
    conversation_id: uuid.UUID,
    turn_id: uuid.UUID,
    user: CanReadDocuments,
    service: ServiceDep,
) -> ConversationTurnOut:
    """Poll one turn without reloading the whole message history."""
    return await conversation_controller.get_turn(
        workspace_id,
        conversation_id,
        turn_id,
        user,
        service,
    )
