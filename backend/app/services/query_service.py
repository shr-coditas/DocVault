"""Where a question enters the system, and which of the two paths it takes.

The supervised graph in ``app.ai.agent`` answers by default. This module still
owns the linear pipeline it grew out of, and falls back to it when there is no
supervisor or no answerer to run the graph with - a deployment without a
provider key still guards, classifies, retrieves and returns passages.

The linear pipeline, which is also the shape of the graph's deterministic gate:

    guardrail chain      cheap, mechanical. Fails → BLOCK, nothing else runs.
    intent classifier    what the query is. No I/O, no model, no database.
    route                only DOCUMENT_QUESTION calls SearchService.

**The ordering is the feature.** Retrieval is the expensive step - an embedding
inference plus an HNSW scan against a permission predicate - and three of the
four intents cannot possibly benefit from it. Classifying first means a greeting,
an out-of-scope request, and an injection attempt each cost a few microseconds of
regex instead. Classify *after* retrieving and the work is already spent by the
time you learn it was pointless.

It is worth being precise about what this does and does not save. The vector
query is one indexed scan; the embedding call is local ONNX inference. Neither is
catastrophic on its own. What the gate really buys is:

- the saving scales with abuse - a caller hammering injections gets rejected at
  regex cost, not inference cost, so the cheap attack stays cheap to refuse;
- the same branch skips the *LLM* call, which is the expensive one and the one
  that is billed;
- a blocked query never reaches the index at all, so there is no window in which
  a hostile prompt has been anywhere near retrieved document text.

Nothing here weakens authorization. Retrieval, when it happens, goes through
``SearchService`` exactly as ``/search`` does, with the same access filter - the
intent gate decides *whether* to search, never *what may be seen*.
"""

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from langchain_core.runnables import Runnable

from app.ai import prompts
from app.config import Settings, get_settings
from app.models.user import User
from app.services.ai_types import (
    ContextReason,
    ContextResolution,
    GeneratedAnswer,
    GuardrailOutcome,
    QueryDecision,
    QueryExecutionContext,
    QueryIntent,
    QueryOutcome,
    SearchHit,
    SearchMode,
)
from app.services.answer_service import AnswerService
from app.services.contextual_query_service import (
    ContextualQueryResolver,
    ResolverUnavailableError,
)
from app.services.guardrail_service import GuardrailService
from app.services.intent_service import IntentClassifier, RuleBasedIntentClassifier
from app.services.search_service import SearchService

logger = structlog.stdlib.get_logger("docvault.query")

# The fixed refusal sentences live in `app.ai.prompts` with the rest of the
# user-facing text; they are re-exported here because this module was where they
# used to be, and callers still import them from it.
DECLINE_MESSAGE = prompts.DECLINE_MESSAGE
BLOCK_MESSAGE = prompts.BLOCK_MESSAGE
CHITCHAT_MESSAGE = prompts.CHITCHAT_MESSAGE
CLARIFICATION_MESSAGE = prompts.CLARIFICATION_MESSAGE
SCOPE_UNAVAILABLE_MESSAGE = prompts.SCOPE_UNAVAILABLE_MESSAGE

QueryContextLoader = Callable[[], Awaitable[QueryExecutionContext]]

# Which intents justify spending a retrieval. A one-line policy table beats the
# same knowledge spread across an if/elif chain, and it is the thing to read when
# asking "why did this query not search?".
_DECISIONS: dict[QueryIntent, QueryDecision] = {
    QueryIntent.DOCUMENT_QUESTION: QueryDecision.RETRIEVE,
    QueryIntent.CHITCHAT: QueryDecision.ANSWER_DIRECTLY,
    QueryIntent.OUT_OF_SCOPE: QueryDecision.DECLINE,
    QueryIntent.PROMPT_INJECTION: QueryDecision.BLOCK,
}

_MESSAGES: dict[QueryDecision, str | None] = {
    # not a fixed sentence: a retrieval's message depends on what came back, so
    # `_retrieval_message` decides it. Null here means "ask that function".
    QueryDecision.RETRIEVE: None,
    QueryDecision.ANSWER_DIRECTLY: CHITCHAT_MESSAGE,
    QueryDecision.DECLINE: DECLINE_MESSAGE,
    QueryDecision.BLOCK: BLOCK_MESSAGE,
    QueryDecision.CLARIFY: CLARIFICATION_MESSAGE,
    QueryDecision.SCOPE_UNAVAILABLE: SCOPE_UNAVAILABLE_MESSAGE,
}


class QueryService:
    def __init__(
        self,
        search: SearchService,
        guardrails: GuardrailService | None = None,
        classifier: IntentClassifier | None = None,
        answers: AnswerService | None = None,
        resolver: ContextualQueryResolver | None = None,
        supervisor: Runnable[Any, Any] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.search = search
        self.guardrails = guardrails or GuardrailService(settings=self.settings)
        self.classifier = classifier or RuleBasedIntentClassifier()
        # Optional: with no answerer the pipeline still guards, classifies, and
        # retrieves - it just returns passages instead of prose. That is the
        # shape a deployment with no API key runs in, and it is a degradation
        # rather than an outage.
        self.answers = answers
        self.resolver = resolver
        # Used only by the graph. The linear pipeline needs no supervision: it
        # searches once and generates once, and there is nothing to decide.
        self.supervisor = supervisor

    async def handle(
        self,
        actor: User,
        workspace_id: uuid.UUID,
        query: str,
        *,
        limit: int | None = None,
        semantic_min_score: float | None = None,
        mode: SearchMode = SearchMode.HYBRID,
        document_id: uuid.UUID | None = None,
        document_ids: list[uuid.UUID] | None = None,
        context_loader: QueryContextLoader | None = None,
    ) -> QueryOutcome:
        """Run the supervised graph, or the linear pipeline it replaced.

        The graph needs both a supervisor to decide and an answerer to write,
        and a deployment can be missing either. Falling back rather than failing
        keeps a workspace with no provider key usable: it still guards,
        classifies, retrieves, and hands back the passages it found.
        """
        if not self.settings.agent_enabled or self.supervisor is None or self.answers is None:
            return await self._handle_legacy(
                actor,
                workspace_id,
                query,
                limit=limit,
                semantic_min_score=semantic_min_score,
                mode=mode,
                document_id=document_id,
                document_ids=document_ids,
                context_loader=context_loader,
            )

        # Imported here so that API startup, and every request on the linear
        # path, gets nowhere near LangGraph.
        from app.ai.agent import AgentContext, run_workflow

        # The conversation's scope and history are resolved before the graph
        # starts rather than inside it. History is state, not a step, and the
        # authorization work behind it - which pinned documents this actor can
        # still open, which earlier turns cited only documents they can still
        # read - belongs to the layer that owns the session, not to a node a
        # model can route around.
        context = QueryExecutionContext(
            document_ids=tuple(document_ids) if document_ids is not None else None
        )
        if context_loader is not None:
            context = await context_loader()
            document_id = None

        return await run_workflow(
            query,
            context.history,
            AgentContext(
                actor=actor,
                workspace_id=workspace_id,
                search=self.search,
                answers=self.answers,
                supervisor=self.supervisor,
                settings=self.settings,
                document_ids=context.document_ids,
                unavailable_documents=context.unavailable_documents,
                document_id=document_id,
                limit=limit,
                semantic_min_score=semantic_min_score,
                mode=mode,
            ),
        )

    async def _handle_legacy(
        self,
        actor: User,
        workspace_id: uuid.UUID,
        query: str,
        *,
        limit: int | None = None,
        semantic_min_score: float | None = None,
        mode: SearchMode = SearchMode.HYBRID,
        document_id: uuid.UUID | None = None,
        document_ids: list[uuid.UUID] | None = None,
        context_loader: QueryContextLoader | None = None,
    ) -> QueryOutcome:
        """Guardrail, classify, and retrieve only if the intent warrants it."""
        guardrails = await self.guardrails.run(query)
        if not guardrails.passed:
            failure = guardrails.failure
            assert failure is not None  # `passed` is false, so one verdict failed
            return self._refused(
                query,
                # a guardrail failure is not a claim about what the query meant;
                # it never got far enough to be classified
                intent=QueryIntent.OUT_OF_SCOPE,
                confidence=1.0,
                decision=QueryDecision.BLOCK,
                reason=f"{failure.name}: {failure.detail}",
                guardrails=guardrails,
                actor=actor,
                workspace_id=workspace_id,
            )

        judgement = await self.classifier.classify(query)
        decision = _DECISIONS[judgement.intent]

        if decision is not QueryDecision.RETRIEVE:
            return self._refused(
                query,
                intent=judgement.intent,
                confidence=judgement.confidence,
                decision=decision,
                reason=judgement.reason,
                guardrails=guardrails,
                actor=actor,
                workspace_id=workspace_id,
            )

        context = QueryExecutionContext(
            document_ids=tuple(document_ids) if document_ids is not None else None
        )
        if context_loader is not None:
            # Conversation scope and history are not even read until the raw
            # current message has passed both safety and intent gates.
            context = await context_loader()
            document_id = None
            document_ids = list(context.document_ids) if context.document_ids is not None else None
            if context.document_ids == ():
                return self._refused(
                    query,
                    intent=judgement.intent,
                    confidence=judgement.confidence,
                    decision=QueryDecision.SCOPE_UNAVAILABLE,
                    reason="selected_scope_empty",
                    guardrails=guardrails,
                    actor=actor,
                    workspace_id=workspace_id,
                    context=context,
                )

        effective_query = query
        resolution: ContextResolution | None = None
        if self.resolver is not None and context.history:
            try:
                resolution = await self.resolver.resolve(query, context.history)
            except ResolverUnavailableError as exc:
                logger.warning(
                    "context_resolution_unavailable",
                    workspace_id=str(workspace_id),
                    actor_id=str(actor.id),
                    failure_type=type(exc).__name__,
                    history_turns=len(context.history),
                    query_chars=len(query),
                )
                resolution = ContextResolution(
                    standalone_query=query,
                    used_history=False,
                    needs_clarification=False,
                    reason_code=ContextReason.FALLBACK,
                )
            if resolution.needs_clarification:
                return self._refused(
                    query,
                    intent=judgement.intent,
                    confidence=judgement.confidence,
                    decision=QueryDecision.CLARIFY,
                    reason=resolution.reason_code.value,
                    guardrails=guardrails,
                    actor=actor,
                    workspace_id=workspace_id,
                    context=context,
                    resolution=resolution,
                )
            effective_query = resolution.standalone_query

        if context.scope_degraded and self._mentions_unavailable(effective_query, context):
            return self._refused(
                effective_query,
                intent=judgement.intent,
                confidence=judgement.confidence,
                decision=QueryDecision.SCOPE_UNAVAILABLE,
                reason="unavailable_document_named",
                guardrails=guardrails,
                actor=actor,
                workspace_id=workspace_id,
                context=context,
                resolution=resolution,
            )

        # ...and only here is retrieval spent
        result = await self.search.search(
            actor,
            workspace_id,
            effective_query,
            mode=mode,
            limit=limit,
            semantic_min_score=semantic_min_score,
            document_id=document_id,
            document_ids=document_ids,
        )
        # ...and only now, with permission-filtered passages in hand, does any
        # workspace text leave the building. Three conditions gate the call, and
        # each is a real saving: no answerer configured, nothing retrieved, or
        # the provider unavailable - all return sources without an answer rather
        # than failing the request.
        answer = None
        selected_sources: tuple[SearchHit, ...] = ()
        if self.answers is not None and result.hits:
            attempt = await self.answers.answer(effective_query, result.hits)
            answer = attempt.answer
            selected_sources = attempt.selected_sources

        self._log(
            actor,
            workspace_id,
            query,
            judgement.intent,
            decision,
            retrieved=True,
            hits=len(result.hits),
            reason=judgement.reason,
            answered=answer is not None,
        )
        return QueryOutcome(
            query=effective_query,
            intent=judgement.intent,
            confidence=judgement.confidence,
            decision=decision,
            reason=judgement.reason,
            guardrails=guardrails,
            retrieval_performed=True,
            hits=result.hits,
            message=self._retrieval_message(result.hits, answer),
            answer=answer,
            selected_sources=selected_sources,
            context_resolution=resolution,
            scope_degraded=context.scope_degraded,
            unavailable_documents=context.unavailable_documents,
            # This path drafts at most once and never validates, so the count is
            # simply whether the answerer was reached. Reported rather than left
            # at zero so the two pipelines stay comparable field for field.
            generation_attempts=1 if (self.answers is not None and result.hits) else 0,
        )

    @staticmethod
    def _retrieval_message(
        hits: tuple[SearchHit, ...], answer: GeneratedAnswer | None
    ) -> str | None:
        """Policy text for a retrieval that produced no answer.

        None when an answer exists - the answer speaks for itself, and a canned
        sentence alongside it would just be noise. Otherwise it distinguishes the
        two silences that look identical to a caller but mean opposite things:
        *we searched and found nothing* versus *we found passages but could not
        generate over them*.
        """
        if answer is not None:
            return None
        return prompts.NO_SOURCES_MESSAGE if not hits else prompts.GENERATION_UNAVAILABLE_MESSAGE

    def _refused(
        self,
        query: str,
        *,
        intent: QueryIntent,
        confidence: float,
        decision: QueryDecision,
        reason: str,
        guardrails: GuardrailOutcome,
        actor: User,
        workspace_id: uuid.UUID,
        context: QueryExecutionContext | None = None,
        resolution: ContextResolution | None = None,
    ) -> QueryOutcome:
        """Every path that returns without retrieving.

        One constructor for all three refusal shapes, so a future intent cannot
        accidentally return a ``QueryOutcome`` claiming hits it never fetched.
        """
        self._log(
            actor, workspace_id, query, intent, decision, retrieved=False, hits=0, reason=reason
        )
        execution_context = context or QueryExecutionContext(document_ids=None)
        return QueryOutcome(
            query=query,
            intent=intent,
            confidence=confidence,
            decision=decision,
            reason=reason,
            guardrails=guardrails,
            retrieval_performed=False,
            hits=(),
            message=_MESSAGES[decision],
            context_resolution=resolution,
            scope_degraded=execution_context.scope_degraded,
            unavailable_documents=execution_context.unavailable_documents,
        )

    @staticmethod
    def _mentions_unavailable(query: str, context: QueryExecutionContext) -> bool:
        """Usability heuristic only; retrieval ACL remains authoritative."""
        haystack = " ".join(query.casefold().split())
        for document in context.unavailable_documents:
            candidates = {
                " ".join(document.title.casefold().split()),
                " ".join(document.file_name.rsplit(".", 1)[0].casefold().split()),
            }
            if any(len(candidate) >= 3 and candidate in haystack for candidate in candidates):
                return True
        return False

    def _log(
        self,
        actor: User,
        workspace_id: uuid.UUID,
        query: str,
        intent: QueryIntent,
        decision: QueryDecision,
        *,
        retrieved: bool,
        hits: int,
        reason: str | None = None,
        answered: bool | None = None,
    ) -> None:
        """Log the decision, never the query.

        A blocked query is the one case where logging the text would genuinely
        help an investigation, and it is also the case where the text is most
        likely to be hostile - logs get read by tools, shipped to third parties,
        and rendered in dashboards. The reason code and the length carry enough
        to spot a pattern without turning the log into an injection vector.
        """
        logger.info(
            "query",
            workspace_id=str(workspace_id),
            actor_id=str(actor.id),
            intent=intent.value,
            decision=decision.value,
            retrieval_performed=retrieved,
            hits=hits,
            reason=reason,
            # None on every path that never retrieved: generation was not in
            # play at all, which is a different fact from "was in play and
            # produced nothing". Same reason Completion's token counts are
            # optional rather than zero.
            answered=answered,
            query_chars=len(query),
        )
