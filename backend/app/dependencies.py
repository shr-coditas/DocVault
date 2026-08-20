"""Shared FastAPI dependencies: DB session, current user, permission guards."""

import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated

import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.session import get_db
from app.exceptions import UnauthorizedError
from app.models.user import User
from app.repository.user_repository import UserRepository
from app.services.contextual_query_service import (
    ContextualQueryResolver,
    get_default_contextual_resolver,
)
from app.services.embedding_service import Embedder, get_default_embedder
from app.services.evidence_grading_service import (
    EvidenceGrader,
    get_default_evidence_grader,
)
from app.services.llm_service import ChatModel, get_default_chat_model
from app.services.output_guardrail_service import (
    DeterministicOutputGuardrail,
    OutputGuardrail,
)
from app.services.permission_service import PermissionService
from app.services.query_analysis_service import (
    QueryAnalyzer,
    get_default_query_analyzer,
)
from app.services.query_planning_service import (
    QueryPlanner,
    get_default_query_planner,
)
from app.services.reranking_service import Reranker, get_default_reranker
from app.services.storage_service import StorageService
from app.utils.rbac_catalog import Perm
from app.utils.security import decode_access_token

DbSession = Annotated[AsyncSession, Depends(get_db)]


def get_storage_service() -> StorageService:
    """Object-storage seam: tests override this to point at a throwaway MinIO."""
    return StorageService()


StorageDep = Annotated[StorageService, Depends(get_storage_service)]


def get_embedder() -> Embedder:
    """Embedding seam: tests override this with a deterministic in-process fake,
    so the suite needs no model download and no network."""
    return get_default_embedder()


EmbedderDep = Annotated[Embedder, Depends(get_embedder)]


def get_reranker() -> Reranker:
    return get_default_reranker()


RerankerDep = Annotated[Reranker, Depends(get_reranker)]


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


def get_evidence_grader() -> EvidenceGrader:
    """Corrective-retrieval grading seam (6C); its own override point.

    Only the graph consults it, so a deployment with the rollout flag off builds
    this object and never calls it. Construction stays free either way: the
    structured model resolves its provider client on first use, not here.
    """
    return get_default_evidence_grader()


EvidenceGraderDep = Annotated[EvidenceGrader, Depends(get_evidence_grader)]


def get_query_analyzer() -> QueryAnalyzer:
    """Structured safety/intent/task analysis seam (6D).

    Layered rather than model-only: the returned analyzer answers a recognized
    attack from the existing rules with no provider call at all, and falls back
    to those same rules when the provider is unavailable.
    """
    return get_default_query_analyzer()


QueryAnalyzerDep = Annotated[QueryAnalyzer, Depends(get_query_analyzer)]


def get_query_planner() -> QueryPlanner:
    """Bounded retrieval-planning seam (6D); graph-only, like the grader."""
    return get_default_query_planner()


QueryPlannerDep = Annotated[QueryPlanner, Depends(get_query_planner)]


def get_output_guardrail() -> OutputGuardrail:
    """Deterministic output checks (6E).

    No provider and no network: these are regex, unicode-category and overlap
    checks, so unlike the other agent seams this one cannot degrade. That is
    deliberate - the last gate before a user sees generated text should not have
    an outage mode that fails open.
    """
    return DeterministicOutputGuardrail()


OutputGuardrailDep = Annotated[OutputGuardrail, Depends(get_output_guardrail)]

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
