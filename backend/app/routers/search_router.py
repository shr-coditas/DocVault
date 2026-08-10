import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.controller.search_controller import search_controller
from app.controller.search_controller.dto.search_dto import SearchResultOut
from app.dependencies import DbSession, EmbedderDep, RerankerDep, require_permission
from app.exceptions import UnprocessableEntityError
from app.models.user import User
from app.services.ai_types import SearchMode
from app.services.search_service import MAX_DOCUMENT_IDS, MAX_LIMIT, SearchService
from app.utils.rbac_catalog import Perm

router = APIRouter(prefix="/workspaces/{workspace_id}/search", tags=["search"])


def get_search_service(
    db: DbSession, embedder: EmbedderDep, reranker: RerankerDep
) -> SearchService:
    return SearchService(db, embedder, reranker=reranker)


ServiceDep = Annotated[SearchService, Depends(get_search_service)]
CanRead = Annotated[User, Depends(require_permission(Perm.DOCUMENT_READ))]


@router.get("")
async def search(
    workspace_id: uuid.UUID,
    user: CanRead,
    service: ServiceDep,
    q: Annotated[str, Query(min_length=1, max_length=1000)],
    mode: Annotated[SearchMode, Query()] = SearchMode.HYBRID,
    limit: Annotated[int | None, Query(ge=1, le=MAX_LIMIT)] = None,
    semantic_min_score: Annotated[float | None, Query(ge=-1.0, le=1.0)] = None,
    document_id: Annotated[uuid.UUID | None, Query()] = None,
    document_ids: Annotated[list[uuid.UUID] | None, Query(max_length=MAX_DOCUMENT_IDS)] = None,
    removed_min_score: Annotated[
        float | None, Query(alias="min_score", include_in_schema=False)
    ] = None,
) -> SearchResultOut:
    if removed_min_score is not None:
        raise UnprocessableEntityError("min_score has been removed; use semantic_min_score")
    if document_id is not None and document_ids:
        raise UnprocessableEntityError("document_id and document_ids are mutually exclusive")
    if mode is SearchMode.LEXICAL and semantic_min_score is not None:
        raise UnprocessableEntityError("semantic score parameters do not apply to lexical mode")
    return await search_controller.search_documents(
        workspace_id,
        q,
        mode,
        limit,
        semantic_min_score,
        document_id,
        document_ids,
        user,
        service,
    )
