"""Transient state contracts for one query-graph invocation."""

from typing import NotRequired, TypedDict

from app.services.ai_types import (
    AnswerDraft,
    ContextResolution,
    EvidenceGrade,
    GeneratedAnswer,
    GuardrailOutcome,
    OutputVerdict,
    QueryAnalysis,
    QueryDecision,
    QueryIntent,
    QueryOutcome,
    QueryPlan,
    QueryTask,
    SearchHit,
)


class QueryGraphInput(TypedDict):
    raw_query: str


class QueryGraphState(QueryGraphInput):
    """Bounded control state; trusted authorization stays in runtime context."""

    guardrails: NotRequired[GuardrailOutcome]
    intent: NotRequired[QueryIntent]
    confidence: NotRequired[float]
    decision: NotRequired[QueryDecision]
    reason: NotRequired[str]
    effective_query: NotRequired[str] # The query that was actually used for retrieval, which may differ from the raw query due to guardrail or intent rewriting.
    context_resolution: NotRequired[ContextResolution | None] # The result of resolving the query's context, which may be None if no context was resolved or if the resolution failed.
    # Bounded planning (6D). ``task`` is what the question was taken to be; the
    # plan holds only search wording and required aspects - never a workspace,
    # an actor, a document id or any other authorization material. ``analysis``
    # is null whenever no structured analyzer ran, which is the 6B/6C shape.
    analysis: NotRequired[QueryAnalysis | None]
    task: NotRequired[QueryTask]
    plan: NotRequired[QueryPlan | None]
    hits: NotRequired[tuple[SearchHit, ...]]
    # Corrective retrieval (6C). ``retrieval_attempts`` bounds the loop, and
    # ``retry_retrieval`` is decided by the grade node because conditional edges
    # see state only - the attempt budget lives in trusted runtime context.
    retrieval_attempts: NotRequired[int]
    grade: NotRequired[EvidenceGrade | None]
    retry_retrieval: NotRequired[bool]
    # Output validation (6E). ``draft`` is unvalidated model text and is cleared
    # the moment it is rejected, so no path can carry it into the outcome, the
    # ledger or a log. ``regenerate`` mirrors ``retry_retrieval``: the budget is
    # runtime-owned, so the node computes the verdict and the edge only reads it.
    draft: NotRequired[AnswerDraft | None]
    generation_attempts: NotRequired[int]
    output_verdict: NotRequired[OutputVerdict | None]
    regenerate: NotRequired[bool]
    answer: NotRequired[GeneratedAnswer | None]
    selected_sources: NotRequired[tuple[SearchHit, ...]]
    outcome: NotRequired[QueryOutcome]


class QueryGraphUpdate(TypedDict, total=False):
    guardrails: GuardrailOutcome
    intent: QueryIntent
    confidence: float
    decision: QueryDecision
    reason: str
    effective_query: str
    context_resolution: ContextResolution | None
    analysis: QueryAnalysis | None
    task: QueryTask
    plan: QueryPlan | None
    hits: tuple[SearchHit, ...]
    retrieval_attempts: int
    grade: EvidenceGrade | None
    retry_retrieval: bool
    draft: AnswerDraft | None
    generation_attempts: int
    output_verdict: OutputVerdict | None
    regenerate: bool
    answer: GeneratedAnswer | None
    selected_sources: tuple[SearchHit, ...]
    outcome: QueryOutcome


class QueryGraphOutput(TypedDict):
    outcome: QueryOutcome
