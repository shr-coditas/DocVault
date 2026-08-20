"""The nodes of one query turn. Four are deterministic; one asks a model."""

import uuid
from collections.abc import Sequence
from typing import Literal

import structlog
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.ai import prompts
from app.ai.agent.context import AgentContext
from app.ai.agent.state import QueryState
from app.ai.agent.supervisor import Supervision, build_brief
from app.config import Settings
from app.services.ai_types import (
    ContextReason,
    ContextResolution,
    GuardrailOutcome,
    QueryDecision,
    QueryIntent,
    QueryOutcome,
    SearchHit,
    UnavailableDocument,
)
from app.services.guardrail_service import GuardrailService
from app.services.intent_service import RuleBasedIntentClassifier
from app.services.output_guardrail_service import check_output

logger = structlog.stdlib.get_logger("docvault.query")

# Stateless: the same compiled patterns for every request.
RULES = RuleBasedIntentClassifier()

# Which readings of a message are decidable from the text alone, and what each
# one costs. Only a document question is worth waking the supervisor for.
RULE_DECISIONS: dict[QueryIntent, QueryDecision] = {
    QueryIntent.DOCUMENT_QUESTION: QueryDecision.RETRIEVE,
    QueryIntent.CHITCHAT: QueryDecision.ANSWER_DIRECTLY,
    QueryIntent.OUT_OF_SCOPE: QueryDecision.DECLINE,
    QueryIntent.PROMPT_INJECTION: QueryDecision.BLOCK,
}


async def screen(
    state: QueryState,
    runtime: Runtime[AgentContext],
) -> Command[Literal["supervise", "respond"]]:
    """Everything that can be settled without a model, settled first.

    Three of the four things this catches - an empty or oversized message, a
    known instruction-override phrasing, a greeting - are decidable from the
    text in microseconds. Deciding them here means a caller hammering injections
    is refused at regex cost rather than at inference cost, and it means no
    hostile message is ever put in front of the supervisor.

    Safety in particular stays here on purpose. The supervisor can decline a
    message it reads as off-topic, but it cannot be talked into *allowing* one
    of these, because it never sees them.
    """
    context = runtime.context
    question = str(state["messages"][-1].content)

    guardrails = await GuardrailService(settings=context.settings).run(question)
    if not guardrails.passed:
        failure = guardrails.failure
        assert failure is not None  # `passed` is false, so one verdict failed
        return Command(
            update=_screened(
                question,
                guardrails,
                # A guardrail failure is not a claim about what the message
                # meant. It never got far enough to be read.
                intent=QueryIntent.OUT_OF_SCOPE,
                confidence=1.0,
                decision=QueryDecision.BLOCK,
                reason=f"{failure.name}: {failure.detail}",
            ),
            goto="respond",
        )

    judgement = await RULES.classify(question)
    decision = RULE_DECISIONS[judgement.intent]
    if decision is not QueryDecision.RETRIEVE:
        return Command(
            update=_screened(
                question,
                guardrails,
                intent=judgement.intent,
                confidence=judgement.confidence,
                decision=decision,
                reason=judgement.reason,
            ),
            goto="respond",
        )

    scope_failure = _scope_failure(question, context)
    if scope_failure is not None:
        return Command(
            update=_screened(
                question,
                guardrails,
                intent=judgement.intent,
                confidence=judgement.confidence,
                decision=QueryDecision.SCOPE_UNAVAILABLE,
                reason=scope_failure,
            ),
            goto="respond",
        )

    return Command(
        update=_screened(
            question,
            guardrails,
            intent=judgement.intent,
            confidence=judgement.confidence,
            decision=decision,
            reason=judgement.reason,
        ),
        goto="supervise",
    )


async def supervise(
    state: QueryState,
    runtime: Runtime[AgentContext],
) -> Command[Literal["retrieve", "write", "respond"]]:
    """Read the conversation and what has been found, and pick the next step.

    This runs once per pass around the loop, so a two-search turn asks three
    times: search, search again, answer. Each call sees the same brief the last
    one did plus whatever came back, which is what lets one decision replace the
    separate analyze, plan and grade calls the previous design made.
    """
    context = runtime.context
    settings = context.settings
    sources = state.get("sources", ())
    searches_run = state.get("searches_run", 0)
    searches_left = max(0, settings.agent_max_searches - searches_run)
    verdict = state.get("verdict")

    brief = build_brief(
        state["question"],
        # Everything before the message being answered. The supervisor uses it
        # only to resolve references; it is quoted data in the brief.
        state["messages"][:-1],
        sources,
        searches_run=searches_run,
        searches_left=searches_left,
        already_searched=state.get("searched", ()),
        rejected_issues=[issue.value for issue in verdict.issues] if verdict is not None else [],
    )

    try:
        decision = await context.supervisor.decide(brief)
    except Exception as exc:
        # One model call, one place that copes with it losing. An outage must not
        # cost the user their turn, so fall back to what an unsupervised pipeline
        # would have done: search once for what they asked, then answer over it.
        logger.warning(
            "supervisor_unavailable",
            workspace_id=str(context.workspace_id),
            actor_id=str(context.actor.id),
            failure_type=type(exc).__name__,
            searches_run=searches_run,
        )
        decision = Supervision(
            action="search" if searches_left and not sources else "answer",
            searches=[state["question"]] if searches_left and not sources else [],
            reason="supervisor_unavailable",
        )

    decision = _within_budget(decision, state, settings)
    question = decision.question.strip() or state["question"]
    update: dict[str, object] = {"reason": decision.reason}
    if question != state["question"]:
        update["question"] = question
        update["resolved_from_history"] = True

    logger.info(
        "supervisor_decision",
        workspace_id=str(context.workspace_id),
        actor_id=str(context.actor.id),
        action=decision.action,
        reason=decision.reason,
        searches_run=searches_run,
        sources=len(sources),
    )

    if decision.action == "search":
        return Command(update={**update, "searches": tuple(decision.searches)}, goto="retrieve")
    if decision.action == "answer":
        return Command(update=update, goto="write")
    if decision.action == "clarify":
        return Command(update={**update, "decision": QueryDecision.CLARIFY}, goto="respond")
    if decision.action == "refuse":
        return Command(
            update={
                **update,
                "intent": QueryIntent.OUT_OF_SCOPE,
                "decision": QueryDecision.DECLINE,
            },
            goto="respond",
        )
    return Command(update={**update, "unsupported": True}, goto="respond")


async def retrieve(
    state: QueryState,
    runtime: Runtime[AgentContext],
) -> Command[Literal["supervise"]]:
    """Run the searches the supervisor asked for, then report back.

    They run one after another. A request-scoped ``AsyncSession`` cannot execute
    statements concurrently, so fanning these out would be a concurrency bug
    rather than a speed-up; real parallelism needs a session per branch, and
    that is worth doing only once these searches are measured.
    """
    context = runtime.context
    merged = state.get("sources", ())
    for query in state.get("searches", ()):
        result = await context.search.search(
            context.actor,
            context.workspace_id,
            query,
            mode=context.mode,
            limit=context.limit,
            semantic_min_score=context.semantic_min_score,
            document_id=context.document_id,
            document_ids=(list(context.document_ids) if context.document_ids is not None else None),
        )
        merged = _merge(merged, result.hits, limit=context.limit)
    return Command(
        update={
            "sources": merged,
            "searches": (),
            "searched": (*state.get("searched", ()), *state.get("searches", ())),
            "searches_run": state.get("searches_run", 0) + 1,
        },
        goto="supervise",
    )


async def write(
    state: QueryState,
    runtime: Runtime[AgentContext],
) -> Command[Literal["supervise", "respond"]]:
    """Draft over the retrieved sources, check the raw text, then cite it.

    The order is the point. Citation resolution *drops* markers that point at no
    supplied source, so checking afterwards would inspect a tidy bibliography
    and never learn that the model invented ``[7]`` - which is the strongest
    single signal that a draft is ungrounded.
    """
    context = runtime.context
    verdict = state.get("verdict")
    attempt = await context.answers.draft(
        state["question"],
        state.get("sources", ()),
        # Only ever one of the fixed corrective sentences, naming the rule that
        # failed. The rejected draft is never quoted back into the prompt.
        guidance=prompts.build_retry_guidance(verdict.issues) if verdict is not None else None,
    )
    written = state.get("drafts_written", 0) + 1
    if attempt.draft is None:
        # The provider was unavailable, or nothing fit the source budget.
        # `respond` reports which; there is nothing to check.
        return Command(
            update={
                "draft": None,
                "answer": None,
                "supplied": attempt.selected_sources,
                "drafts_written": written,
            },
            goto="respond",
        )

    checked = check_output(
        attempt.draft.text,
        attempt.selected_sources,
        system_prompt=prompts.SYSTEM_PROMPT,
        settings=context.settings,
    )
    if checked.passed:
        return Command(
            update={
                "answer": context.answers.finalize_draft(attempt.draft, attempt.selected_sources),
                "draft": None,
                "supplied": attempt.selected_sources,
                "verdict": checked,
                "drafts_written": written,
            },
            goto="respond",
        )

    logger.warning(
        "output_rejected",
        workspace_id=str(context.workspace_id),
        actor_id=str(context.actor.id),
        issues=[issue.value for issue in checked.issues],
        security_failure=checked.security_failure,
        attempt=written,
        # The draft's length, never the draft. A rejected generation is the last
        # text that should be copied into a log shipped somewhere else.
        draft_chars=len(attempt.draft.text),
    )
    return Command(
        # Cleared on every rejection path, so nothing downstream - the outcome,
        # the conversation ledger, a log line - can reach the refused text.
        update={
            "draft": None,
            "answer": None,
            "supplied": attempt.selected_sources,
            "verdict": checked,
            "drafts_written": written,
        },
        # A security failure is terminal. A draft that tried to disclose the
        # system prompt has forfeited its turn, and asking the supervisor
        # whether to try again would put that call in the model's hands. A
        # grounding failure is a quality problem, so the supervisor gets to
        # decide whether the sources support a better attempt.
        goto="respond" if checked.security_failure else "supervise",
    )


async def respond(state: QueryState, runtime: Runtime[AgentContext]) -> dict[str, object]:
    """Assemble the turn's outcome and say, once, what happened."""
    context = runtime.context
    decision = state["decision"]
    retrieved = decision is QueryDecision.RETRIEVE
    sources = state.get("sources", ())
    answer = state.get("answer")
    message = _message(state, retrieved=retrieved)

    logger.info(
        "query",
        workspace_id=str(context.workspace_id),
        actor_id=str(context.actor.id),
        intent=state["intent"].value,
        decision=decision.value,
        reason=state["reason"],
        retrieval_performed=retrieved,
        hits=len(sources),
        # None rather than 0 on a turn that never retrieved: generation was not
        # in play at all there, which is a different fact from "was in play and
        # produced nothing".
        searches=state.get("searches_run") if retrieved else None,
        drafts=state.get("drafts_written") if retrieved else None,
        answered=answer is not None if retrieved else None,
        # The length, never the text. See `write`.
        question_chars=len(state["question"]),
    )

    outcome = QueryOutcome(
        query=state["question"],
        intent=state["intent"],
        confidence=state["confidence"],
        decision=decision,
        reason=state["reason"],
        guardrails=state["guardrails"],
        retrieval_performed=retrieved,
        hits=sources,
        message=message,
        answer=answer,
        selected_sources=state.get("supplied", ()),
        context_resolution=_resolution(state),
        scope_degraded=context.scope_degraded,
        unavailable_documents=context.unavailable_documents,
        evidence_sufficient=_evidence_sufficient(state, retrieved=retrieved),
        output_verdict=state.get("verdict"),
        generation_attempts=state.get("drafts_written", 0),
    )
    # The turn read as a chat: whatever the user is actually shown is appended
    # to the history the next turn will be started with.
    return {
        "outcome": outcome,
        "messages": [AIMessage(content=answer.text if answer is not None else (message or ""))],
    }


def _screened(
    question: str,
    guardrails: GuardrailOutcome,
    *,
    intent: QueryIntent,
    confidence: float,
    decision: QueryDecision,
    reason: str,
) -> dict[str, object]:
    """One shape for everything `screen` decides, however it decided it."""
    return {
        "question": question,
        "guardrails": guardrails,
        "intent": intent,
        "confidence": confidence,
        "decision": decision,
        "reason": reason,
    }


def _scope_failure(question: str, context: AgentContext) -> str | None:
    """Whether this conversation's documents can still answer anything.

    An empty selection means every pinned document has been revoked or deleted:
    retrieving workspace-wide instead would silently answer a question the user
    asked of three specific files. Naming a revoked document is the softer case
    - a usability check, with the retrieval filter still the authority on what
    may actually be read.
    """
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


def _within_budget(
    decision: Supervision,
    state: QueryState,
    settings: Settings,
) -> Supervision:
    """Keep the supervisor's choice inside limits it does not get to set.

    The budgets are configuration, not state, so the supervisor can ask for one
    more search and never for a larger allowance. Reaching a limit is not an
    error: it just means the next best step is the one it would have taken
    anyway once it ran out of searches.
    """
    sources = state.get("sources", ())
    if decision.action == "search":
        # A repeat of a wording already tried returns the same passages, so it
        # is dropped rather than spending a pass to learn that again.
        done = set(state.get("searched", ()))
        wanted = [
            query.strip()
            for query in decision.searches
            if query.strip() and query.strip() not in done
        ]
        exhausted = state.get("searches_run", 0) >= settings.agent_max_searches
        if wanted and not exhausted:
            return decision.model_copy(update={"searches": wanted})
        return decision.model_copy(
            update={
                "action": "answer" if sources else "unsupported",
                "searches": [],
                "reason": "search_budget_spent" if exhausted else "nothing_new_to_search",
            }
        )
    if decision.action == "answer":
        if not sources:
            return decision.model_copy(
                update={"action": "unsupported", "reason": "nothing_retrieved"}
            )
        if state.get("drafts_written", 0) >= settings.agent_max_drafts:
            # The last draft was rejected and there is no attempt left. Ending
            # here rather than answering anyway is the honest outcome: `respond`
            # reports the rejection, not a provider failure.
            return decision.model_copy(
                update={"action": "unsupported", "reason": "draft_budget_spent"}
            )
    return decision


def _merge(
    existing: tuple[SearchHit, ...],
    fresh: tuple[SearchHit, ...],
    *,
    limit: int | None,
) -> tuple[SearchHit, ...]:
    """Union both searches by logical source, best-scoring first, within the limit.

    Chunk ids are not stable enough to deduplicate on: two searches can return
    the same passage under different chunk rows, and the model would then get
    the same text twice under two citation numbers. A passage found by both
    keeps its better score, because the second wording is often the one that
    scores it properly.

    ``limit`` is the caller's source budget, so three searches must not hand the
    answerer three times the passages one search would have.
    """
    best: dict[tuple[uuid.UUID, int, str], SearchHit] = {}
    for hit in (*existing, *fresh):
        identity = (hit.document_id, hit.index_generation, hit.logical_key)
        incumbent = best.get(identity)
        if incumbent is None or hit.score > incumbent.score:
            best[identity] = hit
    ranked = sorted(best.values(), key=lambda hit: hit.score, reverse=True)
    return tuple(ranked if limit is None else ranked[:limit])


def _message(state: QueryState, *, retrieved: bool) -> str | None:
    """What the user is shown when there is no answer to show them.

    A turn that retrieved and produced nothing has five distinct causes that
    look identical from the outside: nothing was found; passages were found and
    do not answer the question; a draft was written and refused on safety
    grounds; a draft was written and could not be kept grounded; or the provider
    could not generate at all. Only the last is an incident, and reporting any
    of the others as one sends a user to check a status page over a working
    system.
    """
    if not retrieved:
        return NON_RETRIEVAL_MESSAGES[state["decision"]]
    if state.get("answer") is not None:
        # The answer speaks for itself; a canned sentence beside it is noise.
        return None
    if not state.get("sources"):
        return prompts.NO_SOURCES_MESSAGE
    verdict = state.get("verdict")
    if verdict is not None and not verdict.passed:
        return (
            prompts.UNSAFE_OUTPUT_MESSAGE
            if verdict.security_failure
            else prompts.REJECTED_ANSWER_MESSAGE
        )
    if state.get("unsupported"):
        return prompts.UNSUPPORTED_EVIDENCE_MESSAGE
    return prompts.GENERATION_UNAVAILABLE_MESSAGE


def _evidence_sufficient(state: QueryState, *, retrieved: bool) -> bool | None:
    """None means nobody judged, which is not the same as judging it thin."""
    if not retrieved or not state.get("sources"):
        return None
    return not state.get("unsupported", False)


def _resolution(state: QueryState) -> ContextResolution | None:
    """How the follow-up was read, reported only when there was a history to read."""
    if len(state["messages"]) < 2:
        return None
    clarifying = state["decision"] is QueryDecision.CLARIFY
    used_history = bool(state.get("resolved_from_history"))
    return ContextResolution(
        standalone_query=state["question"],
        used_history=used_history,
        needs_clarification=clarifying,
        reason_code=(
            ContextReason.AMBIGUOUS
            if clarifying
            else ContextReason.REWRITTEN
            if used_history
            else ContextReason.UNCHANGED
        ),
    )


# Fixed sentences for every ending that is not an answer. Nothing the caller
# typed is echoed back: a refusal that quotes what it refused is how a refusal
# becomes a reflection gadget.
NON_RETRIEVAL_MESSAGES: dict[QueryDecision, str | None] = {
    QueryDecision.RETRIEVE: None,  # `_message` decides; depends on what came back
    QueryDecision.ANSWER_DIRECTLY: prompts.CHITCHAT_MESSAGE,
    QueryDecision.DECLINE: prompts.DECLINE_MESSAGE,
    QueryDecision.BLOCK: prompts.BLOCK_MESSAGE,
    QueryDecision.CLARIFY: prompts.CLARIFICATION_MESSAGE,
    QueryDecision.SCOPE_UNAVAILABLE: prompts.SCOPE_UNAVAILABLE_MESSAGE,
}
