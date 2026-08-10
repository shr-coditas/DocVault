"""The query pipeline: guardrails → intent → (only then) retrieval.

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

import structlog

from app.ai import prompts
from app.models.user import User
from app.services.ai_types import (
    GeneratedAnswer,
    GuardrailOutcome,
    QueryDecision,
    QueryIntent,
    QueryOutcome,
    SearchHit,
    SearchMode,
)
from app.services.answer_service import AnswerService
from app.services.guardrail_service import GuardrailService
from app.services.intent_service import IntentClassifier, RuleBasedIntentClassifier
from app.services.search_service import SearchService

logger = structlog.stdlib.get_logger("docvault.query")

# Fixed sentences, not templates. Nothing the caller typed is echoed back: an
# error message that repeats user input is how a refusal becomes a reflection
# gadget, and there is no reason a refusal needs to quote the thing refused.
DECLINE_MESSAGE = (
    "DocVault answers questions about the documents in this workspace. "
    "That request is outside what it can help with."
)
BLOCK_MESSAGE = "That request was refused."
CHITCHAT_MESSAGE = (
    "Hello. Ask a question about the documents in this workspace and I will look them up."
)

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
}


class QueryService:
    def __init__(
        self,
        search: SearchService,
        guardrails: GuardrailService | None = None,
        classifier: IntentClassifier | None = None,
        answers: AnswerService | None = None,
    ) -> None:
        self.search = search
        self.guardrails = guardrails or GuardrailService()
        self.classifier = classifier or RuleBasedIntentClassifier()
        # Optional: with no answerer the pipeline still guards, classifies, and
        # retrieves - it just returns passages instead of prose. That is the
        # shape a deployment with no API key runs in, and it is a degradation
        # rather than an outage.
        self.answers = answers

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

        # ...and only here is anything spent
        result = await self.search.search(
            actor,
            workspace_id,
            query,
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
        if self.answers is not None and result.hits:
            answer = await self.answers.answer(query, result.hits)

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
            query=query,
            intent=judgement.intent,
            confidence=judgement.confidence,
            decision=decision,
            reason=judgement.reason,
            guardrails=guardrails,
            retrieval_performed=True,
            hits=result.hits,
            message=self._retrieval_message(result.hits, answer),
            answer=answer,
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
    ) -> QueryOutcome:
        """Every path that returns without retrieving.

        One constructor for all three refusal shapes, so a future intent cannot
        accidentally return a ``QueryOutcome`` claiming hits it never fetched.
        """
        self._log(
            actor, workspace_id, query, intent, decision, retrieved=False, hits=0, reason=reason
        )
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
        )

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
