"""Shared FastAPI dependencies: DB session, current user, permission guards."""

import uuid
from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import Annotated, Any

import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.runnables import Runnable
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.agent import create_decision_model
from app.config import get_settings
from app.db.session import get_db
from app.exceptions import UnauthorizedError
from app.models.user import User
from app.repository.user_repository import UserRepository
from app.services.answer_service import CitedAnswerGenerator
from app.services.contextual_query_service import (
    ContextualQueryResolver,
    get_default_contextual_resolver,
)
from app.services.embedding_service import FastEmbedEmbedder, get_default_embedder
from app.services.llm_service import ChatModel, get_default_chat_model
from app.services.permission_service import PermissionService
from app.services.query_service import QueryService
from app.services.reranking_service import FastEmbedReranker, get_default_reranker
from app.services.search_service import SearchService
from app.services.storage_service import StorageService
from app.utils.rbac_catalog import Perm
from app.utils.security import decode_access_token

DbSession = Annotated[AsyncSession, Depends(get_db)]


def get_storage_service() -> StorageService:
    """Object-storage seam: tests override this to point at a throwaway MinIO."""
    return StorageService()


StorageDep = Annotated[StorageService, Depends(get_storage_service)]


def get_embedder() -> FastEmbedEmbedder:
    """Embedding seam: tests override this with a deterministic in-process fake,
    so the suite needs no model download and no network."""
    return get_default_embedder()


EmbedderDep = Annotated[FastEmbedEmbedder, Depends(get_embedder)]


def get_reranker() -> FastEmbedReranker:
    return get_default_reranker()


RerankerDep = Annotated[FastEmbedReranker, Depends(get_reranker)]


def get_chat_model() -> ChatModel:
    """Generation seam, overridden in tests exactly as the embedder is.

    Resolving the singleton here rather than at import time is what lets a
    deployment with no API key still start: ``LangChainChatModel`` builds its
    client on first ``complete()``, so nothing reaches a provider until a
    question actually retrieves something worth answering.
    """
    return get_default_chat_model()


ChatModelDep = Annotated[ChatModel, Depends(get_chat_model)]


def get_contextual_resolver() -> ContextualQueryResolver:
    """Independent follow-up resolver seam; tests override it separately."""
    return get_default_contextual_resolver()


ContextualResolverDep = Annotated[ContextualQueryResolver, Depends(get_contextual_resolver)]


@lru_cache(maxsize=1)
def get_supervisor() -> Runnable[Any, Any]:
    """The chat model the graph's supervisor decides with, shared by requests.

    Cached because it holds a provider client and nothing request-specific;
    building one per request would re-handshake on every question.
    """
    return create_decision_model(get_settings())


SupervisorDep = Annotated[Runnable[Any, Any], Depends(get_supervisor)]


def get_query_service(
    db: DbSession,
    embedder: EmbedderDep,
    reranker: RerankerDep,
    model: ChatModelDep,
    resolver: ContextualResolverDep,
    supervisor: SupervisorDep,
) -> QueryService:
    """Build the one facade shared by legacy and supervisor query execution."""
    return QueryService(
        SearchService(db, embedder, reranker=reranker),
        answers=CitedAnswerGenerator(model),
        resolver=resolver,
        supervisor=supervisor,
    )


QueryServiceDep = Annotated[QueryService, Depends(get_query_service)]

_bearer = HTTPBearer(
    auto_error=False
)  # auto_error= false means that if no credentials are provided, it will return None instead of raising an error. This allows us to handle the case where the user is not authenticated and raise a custom UnauthorizedError.


def _unauthorized() -> UnauthorizedError:
    return UnauthorizedError("not authenticated")


async def get_current_user(
    db: DbSession,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> User:
    if credentials is None:
        raise _unauthorized()

    try:
        payload = decode_access_token(credentials.credentials, get_settings().jwt_secret)
        # a correctly-signed token can still carry a missing or non-uuid `sub`;
        # that is a bad credential (401), not a server fault (500)
        user_id = uuid.UUID(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise _unauthorized() from None

    user = await UserRepository(db).get(user_id)
    if user is None or not user.is_active:
        raise _unauthorized()
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_permission(perm: Perm) -> Callable[..., Awaitable[User]]:
    """Dependency factory: guard an endpoint under /workspaces/{workspace_id}/...

    Resolves the current user, checks the permission against their workspace
    role, and returns the user for use in the handler.
    """

    async def dependency(workspace_id: uuid.UUID, user: CurrentUser, db: DbSession) -> User:
        await PermissionService(db).require(user.id, workspace_id, perm)
        return user

    return dependency
