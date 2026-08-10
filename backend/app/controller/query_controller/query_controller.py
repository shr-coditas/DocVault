import uuid

from app.controller.query_controller.dto.query_dto import (
    GeneratedAnswerOut,
    GuardrailVerdictOut,
    QueryIn,
    QueryOut,
)
from app.controller.search_controller.dto.search_dto import SearchHitOut
from app.models.user import User
from app.services.query_service import QueryService


async def handle_query(
    workspace_id: uuid.UUID,
    data: QueryIn,
    user: User,
    service: QueryService,
) -> QueryOut:
    outcome = await service.handle(
        user,
        workspace_id,
        data.query,
        limit=data.limit,
        semantic_min_score=data.semantic_min_score,
        mode=data.retrieval_mode,
        document_id=data.document_id,
        document_ids=data.document_ids,
    )
    return QueryOut(
        intent=outcome.intent,
        confidence=outcome.confidence,
        decision=outcome.decision,
        reason=outcome.reason,
        guardrails=[
            GuardrailVerdictOut.model_validate(verdict) for verdict in outcome.guardrails.verdicts
        ],
        retrieval_performed=outcome.retrieval_performed,
        hits=[SearchHitOut.model_validate(hit) for hit in outcome.hits],
        message=outcome.message,
        # model_validate walks the nested citations too, so the tuple of
        # frozen dataclasses becomes a list of DTOs without a second loop here
        answer=(
            GeneratedAnswerOut.model_validate(outcome.answer)
            if outcome.answer is not None
            else None
        ),
    )
