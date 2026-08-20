"""Trusted, request-scoped dependencies kept outside model-controlled state."""

import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from app.models.user import User
from app.services.ai_types import QueryDecision, QueryExecutionContext, QueryIntent, SearchMode
from app.services.answer_service import AnswerService
from app.services.contextual_query_service import ContextualQueryResolver
from app.services.evidence_grading_service import EvidenceGrader
from app.services.guardrail_service import GuardrailService
from app.services.intent_service import IntentClassifier
from app.services.output_guardrail_service import OutputGuardrail
from app.services.query_analysis_service import QueryAnalyzer
from app.services.query_planning_service import QueryPlanner
from app.services.search_service import SearchService

# contextloader is invoked by the graph to load a conversation's context, which is then applied to the runtime scope. It is a callable that returns an awaitable QueryExecutionContext, which is the result of loading the context. The loader is optional because some queries may not have a conversation context, and in those cases, the runtime scope will be used as-is.
QueryContextLoader = Callable[[], Awaitable[QueryExecutionContext]]


@dataclass(
    slots=True
)  # dataclass decorator is used to automatically generate special methods like __init__(), __repr__(), and __eq__() for the class, and slots=True is used to optimize memory usage by preventing the creation of a dynamic __dict__ for each instance of the class. slots=True if false, the class will have a __dict__ attribute that allows for dynamic attribute assignment, but it will consume more memory. If slots=True, the class will not have a __dict__ attribute, and only the attributes defined in the class will be allowed, which can save memory when creating many instances of the class. by default the slots attribute is set to False, which means that the class will have a __dict__ attribute and can have dynamic attributes assigned to it. If you set slots=True, the class will not have a __dict__ attribute, and only the attributes defined in the class will be allowed. This can save memory when creating many instances of the class, but it also means that you cannot add new attributes to instances of the class at runtime.
class QueryGraphScope:
    """Authorization-owned effective scope for this invocation.

    A conversation context loader may replace the request's initial scope with
    the currently accessible subset of its immutable selected documents. This
    object is runtime-only so graph or model output cannot widen it.
    """

    document_id: uuid.UUID | None
    document_ids: tuple[uuid.UUID, ...] | None
    execution_context: QueryExecutionContext

    @classmethod
    def from_request(
        cls,
        document_id: uuid.UUID | None,
        document_ids: list[uuid.UUID] | None,
    ) -> "QueryGraphScope":
        frozen_ids = tuple(document_ids) if document_ids is not None else None
        return cls(
            document_id=document_id,
            document_ids=frozen_ids,
            execution_context=QueryExecutionContext(document_ids=frozen_ids),
        )

    def apply_conversation_context(self, context: QueryExecutionContext) -> None:
        self.document_id = None
        self.document_ids = context.document_ids
        self.execution_context = context


@dataclass(slots=True)
class QueryGraphRuntime:
    """Trusted, request-scoped runtime context for one query-graph invocation."""
    actor: User
    workspace_id: uuid.UUID
    search: SearchService
    guardrails: GuardrailService
    classifier: IntentClassifier
    answers: AnswerService | None
    resolver: ContextualQueryResolver | None
    scope: QueryGraphScope
    decisions: Mapping[QueryIntent, QueryDecision]
    messages: Mapping[QueryDecision, str | None]
    context_loader: QueryContextLoader | None = None
    limit: int | None = None
    semantic_min_score: float | None = None
    mode: SearchMode = SearchMode.HYBRID
    # Corrective retrieval (6C). No grader means no grading and no retry, which
    # is exactly the 6B parity path. The attempt budget is runtime-owned: a
    # graded verdict may ask for another search, never for a larger allowance.
    grader: EvidenceGrader | None = None
    max_retrieval_attempts: int = 1
    # Structured analysis and bounded planning (6D). Each is independently
    # optional, and absent means the earlier behavior: the rule-based classifier
    # decides intent, and retrieval runs one search for the resolved question.
    analyzer: QueryAnalyzer | None = None
    planner: QueryPlanner | None = None
    max_subqueries: int = 1
    # Output validation (6E). No guardrail means a drafted answer is finalized
    # unchecked, exactly as 6B-6D did. Like every other budget here, the
    # generation allowance is runtime-owned so a rejected draft can ask for one
    # more attempt but never for a larger allowance.
    output_guardrail: OutputGuardrail | None = None
    max_generation_attempts: int = 1
