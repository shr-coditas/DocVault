"""Nodes for the small, fail-open document-question graph."""

import asyncio
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.runtime import Runtime

from app.ai import prompts
from app.ai.agent.llm_response_dto import ScopeDecision
from app.ai.agent.prompt_utils import SUPERVISOR_PROMPT, make_scope_payload
from app.config import Settings
from app.models.user import User
from app.services.ai_types import (
    ContextReason,
    ContextResolution,
    ConversationTurn,
    DocumentBrief,
    GeneratedAnswer,
    GuardrailOutcome,
    OutputVerdict,
    QueryDecision,
    QueryIntent,
    QueryOutcome,
    SearchHit,
    SearchMode,
    UnavailableDocument,
)
from app.services.answer_service import CitedAnswerGenerator
from app.services.contextual_query_service import (
    ContextualQueryResolver,
    ResolverUnavailableError,
)
from app.services.guardrail_service import GuardrailService
from app.services.intent_service import RuleBasedIntentClassifier
from app.services.output_guardrail_service import check_output
from app.services.search_service import SearchService

logger = structlog.stdlib.get_logger("docvault.query")

RULES = RuleBasedIntentClassifier()

RULE_DECISIONS: dict[QueryIntent, QueryDecision] = {
    QueryIntent.DOCUMENT_QUESTION: QueryDecision.RETRIEVE,
    QueryIntent.CHITCHAT: QueryDecision.ANSWER_DIRECTLY,
    QueryIntent.OUT_OF_SCOPE: QueryDecision.DECLINE,
    QueryIntent.PROMPT_INJECTION: QueryDecision.BLOCK,
}

NON_RETRIEVAL_MESSAGES: dict[QueryDecision, str | None] = {
    QueryDecision.RETRIEVE: None,
    QueryDecision.ANSWER_DIRECTLY: prompts.CHITCHAT_MESSAGE,
    QueryDecision.DECLINE: prompts.DECLINE_MESSAGE,
    QueryDecision.BLOCK: prompts.BLOCK_MESSAGE,
    QueryDecision.CLARIFY: prompts.CLARIFICATION_MESSAGE,
    QueryDecision.SCOPE_UNAVAILABLE: prompts.SCOPE_UNAVAILABLE_MESSAGE,
}


@dataclass(slots=True)
class Context:
    """Request-scoped services and authorization-owned retrieval scope."""

    actor: User
    workspace_id: uuid.UUID
    search: SearchService
    answers: CitedAnswerGenerator
    supervisor: Runnable[Any, Any]
    settings: Settings
    document_ids: tuple[uuid.UUID, ...] | None = None
    unavailable_documents: tuple[UnavailableDocument, ...] = ()
    document_id: uuid.UUID | None = None
    limit: int | None = None
    semantic_min_score: float | None = None
    mode: SearchMode = SearchMode.HYBRID
    resolver: ContextualQueryResolver | None = None

    @property
    def scope_degraded(self) -> bool:
        return bool(self.unavailable_documents)


class State(TypedDict):
    question: str
    history: tuple[ConversationTurn, ...]
    document_summaries: tuple[DocumentBrief, ...]
    summaries_complete: bool
    guardrails: NotRequired[GuardrailOutcome]
    intent: NotRequired[QueryIntent]
    confidence: NotRequired[float]
    decision: NotRequired[QueryDecision]
    reason: NotRequired[str]
    message: NotRequired[str]
    next_step: NotRequired[str]
    context_resolution: NotRequired[ContextResolution]
    outcome: NotRequired[QueryOutcome]


async def guard(state: State, runtime: Runtime[Context]) -> dict[str, object]:
    question = state["question"]
    guardrails = await GuardrailService(settings=runtime.context.settings).run(question)
    if guardrails.passed:
        return {"guardrails": guardrails, "next_step": "classify"}

    failure = guardrails.failure
    assert failure is not None
    return {
        "guardrails": guardrails,
        "intent": QueryIntent.OUT_OF_SCOPE,
        "confidence": 1.0,
        "decision": QueryDecision.BLOCK,
        "reason": f"{failure.name}: {failure.detail}",
        "next_step": "respond",
    }


async def classify(state: State, runtime: Runtime[Context]) -> dict[str, object]:
    context = runtime.context
    judgement = await RULES.classify(state["question"])
    decision = RULE_DECISIONS[judgement.intent]
    update: dict[str, object] = {
        "intent": judgement.intent,
        "confidence": judgement.confidence,
        "decision": decision,
        "reason": judgement.reason,
        "next_step": "respond",
    }
    if decision is not QueryDecision.RETRIEVE:
        return update

    scope_failure = _scope_failure(state["question"], context)
    if scope_failure is not None:
        return {
            **update,
            "decision": QueryDecision.SCOPE_UNAVAILABLE,
            "reason": scope_failure,
        }
    return {**update, "next_step": "rephrase"}


async def rephrase(state: State, runtime: Runtime[Context]) -> dict[str, object]:
    history = state["history"]
    resolver = runtime.context.resolver
    if not history or resolver is None:
        return {"next_step": "supervisor"}

    try:
        resolution = await resolver.resolve(state["question"], history)
    except ResolverUnavailableError as exc:
        logger.warning(
            "context_resolution_unavailable",
            workspace_id=str(runtime.context.workspace_id),
            actor_id=str(runtime.context.actor.id),
            failure_type=type(exc).__name__,
            history_turns=len(history),
            query_chars=len(state["question"]),
        )
        resolution = ContextResolution(
            standalone_query=state["question"],
            used_history=False,
            needs_clarification=False,
            reason_code=ContextReason.FALLBACK,
        )

    if resolution.needs_clarification:
        return {
            "context_resolution": resolution,
            "decision": QueryDecision.CLARIFY,
            "reason": resolution.reason_code.value,
            "message": prompts.CLARIFICATION_MESSAGE,
            "next_step": "respond",
        }

    return {
        "question": resolution.standalone_query,
        "context_resolution": resolution,
        "next_step": "supervisor",
    }


async def supervisor(state: State, runtime: Runtime[Context]) -> dict[str, object]:
    """Use complete selected-document summaries only as a weak scope hint."""
    context = runtime.context
    if not state["summaries_complete"] or not state["document_summaries"]:
        return {"reason": "summaries_incomplete", "next_step": "respond"}

    try:
        result = await asyncio.wait_for(
            context.supervisor.ainvoke(
                [
                    SystemMessage(content=SUPERVISOR_PROMPT),
                    HumanMessage(
                        content=make_scope_payload(
                            state["question"],
                            state["document_summaries"],
                        )
                    ),
                ]
            ),
            timeout=context.settings.agent_timeout_seconds,
        )
        if not isinstance(result, ScopeDecision):
            raise TypeError("the supervisor returned an invalid scope decision")
    except Exception as exc:
        logger.warning(
            "supervisor_unavailable",
            workspace_id=str(context.workspace_id),
            actor_id=str(context.actor.id),
            failure_type=type(exc).__name__,
        )
        return {"reason": "supervisor_unavailable", "next_step": "respond"}

    logger.info(
        "supervisor_scope_decision",
        workspace_id=str(context.workspace_id),
        actor_id=str(context.actor.id),
        in_scope=result.in_scope,
        reason=result.reason,
        documents=len(state["document_summaries"]),
    )
    return {
        "reason": result.reason,
        "next_step": "respond" if result.in_scope else "out_of_scope",
    }


async def out_of_scope(state: State, runtime: Runtime[Context]) -> dict[str, object]:
    del runtime
    return {
        "decision": QueryDecision.DECLINE,
        "reason": state["reason"],
        "message": prompts.DECLINE_MESSAGE,
    }


async def respond(state: State, runtime: Runtime[Context]) -> dict[str, object]:
    """Run the one search/one draft path and assemble the final outcome."""
    context = runtime.context
    decision = state["decision"]
    message = state.get("message")
    if message is None and decision is not QueryDecision.RETRIEVE:
        message = NON_RETRIEVAL_MESSAGES[decision]

    hits: tuple[SearchHit, ...] = ()
    selected_sources: tuple[SearchHit, ...] = ()
    answer: GeneratedAnswer | None = None
    verdict: OutputVerdict | None = None
    generation_attempts = 0
    should_retrieve = decision is QueryDecision.RETRIEVE and message is None

    if should_retrieve:
        result = await context.search.search(
            context.actor,
            context.workspace_id,
            state["question"],
            mode=context.mode,
            limit=context.limit,
            semantic_min_score=context.semantic_min_score,
            document_id=context.document_id,
            document_ids=(list(context.document_ids) if context.document_ids is not None else None),
        )
        hits = result.hits
        if not hits:
            message = prompts.NO_SOURCES_MESSAGE
        else:
            generation_attempts = 1
            attempt = await context.answers.draft(state["question"], hits)
            selected_sources = attempt.selected_sources
            if attempt.draft is None:
                message = prompts.GENERATION_UNAVAILABLE_MESSAGE
            else:
                verdict = check_output(
                    attempt.draft.text,
                    selected_sources,
                    system_prompt=prompts.SYSTEM_PROMPT,
                    settings=context.settings,
                )
                if verdict.passed:
                    answer = context.answers.finalize_draft(
                        attempt.draft,
                        selected_sources,
                    )
                elif verdict.security_failure:
                    message = prompts.UNSAFE_OUTPUT_MESSAGE
                else:
                    message = prompts.REJECTED_ANSWER_MESSAGE

    logger.info(
        "query",
        workspace_id=str(context.workspace_id),
        actor_id=str(context.actor.id),
        intent=state["intent"].value,
        decision=decision.value,
        reason=state["reason"],
        retrieval_performed=should_retrieve,
        hits=len(hits),
        answered=answer is not None if should_retrieve else None,
        query_chars=len(state["question"]),
    )
    outcome = QueryOutcome(
        query=state["question"],
        intent=state["intent"],
        confidence=state["confidence"],
        decision=decision,
        reason=state["reason"],
        guardrails=state["guardrails"],
        retrieval_performed=should_retrieve,
        hits=hits,
        message=message,
        answer=answer,
        selected_sources=selected_sources,
        context_resolution=state.get("context_resolution"),
        scope_degraded=context.scope_degraded,
        unavailable_documents=context.unavailable_documents,
        output_verdict=verdict,
        generation_attempts=generation_attempts,
    )
    return {"outcome": outcome}


def _scope_failure(question: str, context: Context) -> str | None:
    if context.document_ids == ():
        return "selected_scope_empty"
    if context.scope_degraded and _names_unavailable(question, context.unavailable_documents):
        return "unavailable_document_named"
    return None


def _names_unavailable(question: str, documents: Sequence[UnavailableDocument]) -> bool:
    haystack = " ".join(question.casefold().split())
    for document in documents:
        candidates = {
            " ".join(document.title.casefold().split()),
            " ".join(document.file_name.rsplit(".", 1)[0].casefold().split()),
        }
        if any(len(candidate) >= 3 and candidate in haystack for candidate in candidates):
            return True
    return False
