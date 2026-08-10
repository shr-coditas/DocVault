import uuid

from app.controller.search_controller.dto.search_dto import SearchHitOut, SearchResultOut
from app.models.user import User
from app.services.ai_types import SearchMode
from app.services.search_service import SearchService


async def search_documents(
    workspace_id: uuid.UUID,
    query: str,
    mode: SearchMode,
    limit: int | None,
    semantic_min_score: float | None,
    document_id: uuid.UUID | None,
    document_ids: list[uuid.UUID] | None,
    user: User,
    service: SearchService,
) -> SearchResultOut:
    result = await service.search(
        user,
        workspace_id,
        query,
        mode=mode,
        limit=limit,
        semantic_min_score=semantic_min_score,
        document_id=document_id,
        document_ids=document_ids,
    )
    return SearchResultOut(
        query=result.query,
        mode=result.mode,
        limit=result.limit,
        semantic_min_score=result.semantic_min_score,
        count=len(result.hits),
        hits=[SearchHitOut.model_validate(hit) for hit in result.hits],
    )


async def search_semantic(
    workspace_id: uuid.UUID,
    query: str,
    limit: int | None,
    semantic_min_score: float | None,
    document_id: uuid.UUID | None,
    user: User,
    service: SearchService,
) -> SearchResultOut:
    return await search_documents(
        workspace_id,
        query,
        SearchMode.SEMANTIC,
        limit,
        semantic_min_score,
        document_id,
        None,
        user,
        service,
    )
