"""Behavioral-parity nodes for one bounded DocVault query turn."""

from __future__ import annotations

import uuid
from dataclasses import replace

import structlog
from langgraph.runtime import Runtime

from app.ai import prompts
from app.ai.query_graph.runtime import QueryGraphRuntime
from app.ai.query_graph.state import QueryGraphState, QueryGraphUpdate
from app.services.ai_types import (
    ContextReason,
    ContextResolution,
    EvidenceGrade,
    GeneratedAnswer,
    QueryAnalysis,
    QueryDecision,
    QueryExecutionContext,
    QueryIntent,
    QueryOutcome,
    QueryPlan,
    QueryTask,
    SafetyVerdict,
    SearchHit,
)
from app.services.contextual_query_service import ResolverUnavailableError
from app.services.structured_model_service import StructuredModelUnavailableError

logger = structlog.stdlib.get_logger("docvault.query")


async def guard_input(
    state: QueryGraphState,
    runtime: Runtime[QueryGraphRuntime],
) -> QueryGraphUpdate:
    guardrails = await runtime.context.guardrails.run(state["raw_query"])
    update = QueryGraphUpdate(guardrails=guardrails)
    if not guardrails.passed:
        failure = guardrails.failure
        assert failure is not None 
        update["intent"] = QueryIntent.OUT_OF_SCOPE
        update["confidence"] = 1.0 # hard-coded because the guardrail is deterministic and the model is not involved in this decision
        update["decision"] = QueryDecision.BLOCK
        update["reason"] = f"{failure.name}: {failure.detail}"
    return update


async def analyze_query(
    state: QueryGraphState,
    runtime: Runtime[QueryGraphRuntime],
) -> QueryGraphUpdate:
    """Decide what the message is and whether it is safe to act on (6D).

    With no analyzer configured this is the 6B node exactly: the rule-based
    classifier decides, and the task is a plain lookup. With one configured,
    ``LayeredQueryAnalyzer`` still answers known attacks deterministically and
    still falls back to those rules when the provider is unavailable, so adding
    a model here cannot make a hostile message *harder* to catch.
    """
    analyzer = runtime.context.analyzer
    if analyzer is None:
        judgement = await runtime.context.classifier.classify(state["raw_query"])
        # `task` is deliberately left unset rather than defaulted to a lookup.
        # It follows the same convention as `plan`, `evidence` and
        # `output_verdict`: null means the step never ran, which is a different
        # fact from "ran and found a lookup", and it keeps this path's outcome
        # identical to the non-graph pipeline's.
        return QueryGraphUpdate(
            intent=judgement.intent,
            confidence=judgement.confidence,
            decision=runtime.context.decisions[judgement.intent],
            reason=judgement.reason,
            analysis=None,
        )

    try:
        analysis = await analyzer.analyze(state["raw_query"])
    except StructuredModelUnavailableError as exc:
        # `LayeredQueryAnalyzer` absorbs this itself; a bare structured analyzer
        # does not. Degrading to the rules rather than propagating matters most
        # here of all nodes: an analyzer outage that raised would take down
        # ordinary questions, and the rules it falls back to are the same ones
        # that were the whole classifier before this slice.
        logger.warning(
            "query_analysis_unavailable",
            workspace_id=str(runtime.context.workspace_id),
            actor_id=str(runtime.context.actor.id),
            failure_type=type(exc).__name__,
        )
        judgement = await runtime.context.classifier.classify(state["raw_query"])
        return QueryGraphUpdate(
            intent=judgement.intent,
            confidence=judgement.confidence,
            decision=runtime.context.decisions[judgement.intent],
            reason=judgement.reason,
            analysis=None,
        )
    # Safety and intent are separate verdicts on purpose: an attempt to extract
    # the system prompt is often a perfectly document-shaped question, and the
    # analyzer reports it as one. Anything other than an explicit allow is
    # reported to the caller as an injection block, which is what the rule-based
    # path already did - `uncertain` is not a licence to retrieve.
    if analysis.safety is not SafetyVerdict.ALLOW:
        logger.info(
            "query_analysis_blocked",
            workspace_id=str(runtime.context.workspace_id),
            actor_id=str(runtime.context.actor.id),
            verdict=analysis.safety.value,
            categories=[category.value for category in analysis.safety_categories],
            reason_code=analysis.reason_code,
        )
        return QueryGraphUpdate(
            intent=QueryIntent.PROMPT_INJECTION,
            confidence=analysis.confidence,
            decision=QueryDecision.BLOCK,
            reason=analysis.reason_code,
            analysis=analysis,
            task=analysis.task,
        )

    return QueryGraphUpdate(
        intent=analysis.intent,
        confidence=analysis.confidence,
        decision=runtime.context.decisions[analysis.intent],
        reason=analysis.reason_code,
        analysis=analysis,
        task=analysis.task,
    )


async def load_context(
    state: QueryGraphState,
    runtime: Runtime[QueryGraphRuntime],
) -> QueryGraphUpdate:
    loader = runtime.context.context_loader
    if loader is None:
        return QueryGraphUpdate()
    context = await loader()
    runtime.context.scope.apply_conversation_context(context)
    if context.document_ids == ():
        return QueryGraphUpdate(
            decision=QueryDecision.SCOPE_UNAVAILABLE,
            reason="selected_scope_empty",
        )
    return QueryGraphUpdate()


async def resolve_context(
    state: QueryGraphState,
    runtime: Runtime[QueryGraphRuntime],
) -> QueryGraphUpdate:
    raw_query = state["raw_query"]
    context = runtime.context.scope.execution_context
    resolution: ContextResolution | None = None
    effective_query = raw_query
    resolver = runtime.context.resolver
    if resolver is not None and context.history:
        try:
            resolution = await resolver.resolve(raw_query, context.history)
        except ResolverUnavailableError as exc:
            logger.warning(
                "context_resolution_unavailable",
                workspace_id=str(runtime.context.workspace_id),
                actor_id=str(runtime.context.actor.id),
                failure_type=type(exc).__name__,
                history_turns=len(context.history),
                query_chars=len(raw_query),
            )
            resolution = ContextResolution(
                standalone_query=raw_query,
                used_history=False,
                needs_clarification=False,
                reason_code=ContextReason.FALLBACK,
            )
        if resolution.needs_clarification:
            return QueryGraphUpdate(
                decision=QueryDecision.CLARIFY,
                reason=resolution.reason_code.value,
                context_resolution=resolution,
            )
        effective_query = resolution.standalone_query

    update = QueryGraphUpdate(
        effective_query=effective_query,
        context_resolution=resolution,
    )
    if context.scope_degraded and _mentions_unavailable(effective_query, context):
        update["decision"] = QueryDecision.SCOPE_UNAVAILABLE
        update["reason"] = "unavailable_document_named"
    return update


async def plan_retrieval(
    state: QueryGraphState,
    runtime: Runtime[QueryGraphRuntime],
) -> QueryGraphUpdate:
    """Decide what to search for, after the question is safe and resolved (6D).

    Planning deliberately runs *after* contextual resolution: a follow-up like
    "what about contractors?" cannot be decomposed sensibly until it has been
    rewritten into a standalone question.

    A plan contains search wording and required aspects, and nothing else. It
    cannot name a workspace, an actor, a document or a filter, because the scope
    those would target lives in runtime context and is applied by `retrieve`
    regardless of what the plan says.
    """
    planner = runtime.context.planner
    if planner is None:
        return QueryGraphUpdate(plan=None)

    query = state.get("effective_query", state["raw_query"])
    try:
        plan = await planner.plan(query, _analysis_for_planning(state, query))
    except StructuredModelUnavailableError as exc:
        # `FallbackQueryPlanner` normally absorbs this; a bare planner may not.
        # Either way an unplanned query is a single-search query, not a failed
        # one - the same degradation grading takes.
        logger.warning(
            "query_planning_unavailable",
            workspace_id=str(runtime.context.workspace_id),
            actor_id=str(runtime.context.actor.id),
            failure_type=type(exc).__name__,
        )
        return QueryGraphUpdate(plan=None)

    if plan.needs_clarification:
        # The planner's own question is deliberately not shown. It is unvalidated
        # model text, and every other terminal message in this graph is a fixed
        # sentence; making clarification the one exception would put a new
        # unchecked output surface in front of the user for no product gain.
        return QueryGraphUpdate(
            plan=plan,
            task=plan.task,
            decision=QueryDecision.CLARIFY,
            reason="plan_clarification",
        )

    queries = plan.search_queries[: runtime.context.max_subqueries]
    logger.info(
        "query_planned",
        workspace_id=str(runtime.context.workspace_id),
        actor_id=str(runtime.context.actor.id),
        task=plan.task.value,
        searches=len(queries),
        required_aspects=len(plan.required_aspects),
    )
    return QueryGraphUpdate(plan=replace(plan, search_queries=queries), task=plan.task)


async def retrieve(
    state: QueryGraphState,
    runtime: Runtime[QueryGraphRuntime],
) -> QueryGraphUpdate:
    """Run the planned searches and merge what this actor is allowed to see.

    The searches run **sequentially**. A request-scoped ``AsyncSession`` cannot
    execute statements in parallel, so a `RunnableParallel` fan-out over the
    caller's session would be a concurrency bug rather than a speed-up; genuine
    parallelism needs an injected factory that opens one read session per branch,
    which is deferred until the retrieval gates are actually measured.
    """
    scope = runtime.context.scope
    merged = state.get("hits", ())
    for query in _search_queries(state, runtime):
        result = await runtime.context.search.search(
            runtime.context.actor,
            runtime.context.workspace_id,
            query,
            mode=runtime.context.mode,
            limit=runtime.context.limit,
            semantic_min_score=runtime.context.semantic_min_score,
            document_id=scope.document_id,
            document_ids=list(scope.document_ids) if scope.document_ids is not None else None,
        )
        # Both a corrective attempt and a sibling subquery add to the evidence
        # rather than replacing it: either may have found a genuinely relevant
        # passage the other missed. Every hit on every branch passed the same
        # permission filter, so merging cannot widen what this actor may see.
        merged = _merge_hits(merged, result.hits, limit=runtime.context.limit)
    return QueryGraphUpdate(
        hits=merged,
        retrieval_attempts=state.get("retrieval_attempts", 0) + 1,
    )


async def grade_evidence(
    state: QueryGraphState,
    runtime: Runtime[QueryGraphRuntime],
) -> QueryGraphUpdate:
    """Judge whether the retrieved passages support an answer (6C).

    With no grader configured this is a no-op and the graph keeps its 6B shape.
    A grading failure is never fatal: an unjudged query proceeds to generation
    exactly as it did before, because a degraded grader must not cost the user
    an answer the sources could already support.
    """
    grader = runtime.context.grader
    if grader is None:
        return QueryGraphUpdate()

    query = state.get("effective_query", state["raw_query"])
    hits = state.get("hits", ())
    # The plan is what the grader measures coverage against: its required
    # aspects are how a comparison question gets judged on whether *both* sides
    # were actually found. With no planner configured the plan describes what
    # was searched, which is the single resolved question.
    plan = state.get("plan") or QueryPlan(
        task=state.get("task", QueryTask.LOOKUP),
        search_queries=(query,),
    )
    try:
        grade = await grader.grade(query, hits, plan)
    except StructuredModelUnavailableError as exc:
        logger.warning(
            "evidence_grading_unavailable",
            workspace_id=str(runtime.context.workspace_id),
            actor_id=str(runtime.context.actor.id),
            failure_type=type(exc).__name__,
            hits=len(hits),
        )
        return QueryGraphUpdate(grade=None, retry_retrieval=False)

    return QueryGraphUpdate(grade=grade, retry_retrieval=_should_retry(state, runtime, grade))


async def rewrite_query(
    state: QueryGraphState,
    runtime: Runtime[QueryGraphRuntime],
) -> QueryGraphUpdate:
    """Adopt the grader's suggested wording for one more search.

    Only the search *wording* changes. Actor, workspace and document scope stay
    in runtime context, so a rewrite can look elsewhere in what this actor may
    already read - never at anything more.
    """
    grade = state.get("grade")
    assert grade is not None and grade.suggested_query is not None  # _should_retry checked both
    logger.info(
        "corrective_retrieval_rewrite",
        workspace_id=str(runtime.context.workspace_id),
        actor_id=str(runtime.context.actor.id),
        attempt=state.get("retrieval_attempts", 0),
        missing_aspects=len(grade.missing_aspects),
        query_chars=len(grade.suggested_query),
    )
    return QueryGraphUpdate(effective_query=grade.suggested_query, retry_retrieval=False)


async def generate(
    state: QueryGraphState,
    runtime: Runtime[QueryGraphRuntime],
) -> QueryGraphUpdate:
    """Draft an answer over the merged evidence, without resolving citations.

    Citations are resolved in `validate_output`, once the raw text has been
    checked - resolving first would quietly delete the invented markers that are
    the strongest evidence a draft is ungrounded.
    """
    hits = state["hits"]
    # The corrective loop is over by the time this runs. Evidence still judged
    # insufficient is a terminal answer in itself: generating anyway would spend
    # the billed call to produce a paragraph the sources cannot support.
    if _evidence_insufficient(state):
        logger.info(
            "insufficient_evidence_after_correction",
            workspace_id=str(runtime.context.workspace_id),
            actor_id=str(runtime.context.actor.id),
            attempts=state.get("retrieval_attempts", 0),
            hits=len(hits),
        )
        return QueryGraphUpdate(draft=None, answer=None, selected_sources=())
    if runtime.context.answers is None or not hits:
        return QueryGraphUpdate(draft=None, answer=None, selected_sources=())

    verdict = state.get("output_verdict")
    attempt = await runtime.context.answers.draft(
        state.get("effective_query", state["raw_query"]),
        hits,
        # Only ever a fixed sentence naming the rule that failed. The rejected
        # draft is never quoted back into the prompt.
        guidance=prompts.build_retry_guidance(verdict.issues) if verdict is not None else None,
    )
    return QueryGraphUpdate(
        draft=attempt.draft,
        selected_sources=attempt.selected_sources,
        generation_attempts=state.get("generation_attempts", 0) + 1,
    )


async def validate_output(
    state: QueryGraphState,
    runtime: Runtime[QueryGraphRuntime],
) -> QueryGraphUpdate:
    """Check the draft before anyone sees it, then resolve its citations (6E).

    Ordering is the whole point. A guardrail that runs after delivery cannot
    retract bytes already sent, and one that runs after citation resolution
    inspects a tidied bibliography rather than what the model actually wrote.

    Two failure classes, deliberately not treated alike: a security failure is
    terminal, because a draft that tried to disclose the prompt has forfeited
    its turn; a grounding or citation failure is a quality problem and earns one
    corrective retry, because the sources may well support a better answer.
    """
    draft = state.get("draft")
    answers = runtime.context.answers
    sources = state.get("selected_sources", ())
    if draft is None or answers is None:
        # Nothing was generated: no sources, no answerer, provider outage, or
        # evidence graded insufficient. `finalize` already reports each of those.
        return QueryGraphUpdate(regenerate=False)

    guardrail = runtime.context.output_guardrail
    if guardrail is None:
        return QueryGraphUpdate(
            answer=answers.finalize_draft(draft, sources),
            draft=None,
            regenerate=False,
        )

    verdict = await guardrail.validate(draft.text, sources, system_prompt=prompts.SYSTEM_PROMPT)
    if verdict.passed:
        return QueryGraphUpdate(
            answer=answers.finalize_draft(draft, sources),
            draft=None,
            output_verdict=verdict,
            regenerate=False,
        )

    regenerate = verdict.can_regenerate and (
        state.get("generation_attempts", 0) < runtime.context.max_generation_attempts
    )
    logger.warning(
        "output_rejected",
        workspace_id=str(runtime.context.workspace_id),
        actor_id=str(runtime.context.actor.id),
        issues=[issue.value for issue in verdict.issues],
        security_failure=verdict.security_failure,
        attempt=state.get("generation_attempts", 0),
        regenerating=regenerate,
        # the draft's length, never the draft: a rejected generation is the last
        # text that should be copied into a log shipped somewhere else
        draft_chars=len(draft.text),
    )
    # `draft` is cleared on every rejection path, so no later node, outcome,
    # ledger row or log statement can reach the text that was refused.
    return QueryGraphUpdate(
        draft=None,
        answer=None,
        output_verdict=verdict,
        regenerate=regenerate,
    )


async def finalize(
    state: QueryGraphState,
    runtime: Runtime[QueryGraphRuntime],
) -> QueryGraphUpdate:
    query = state.get("effective_query", state["raw_query"])
    decision = state["decision"]
    context = runtime.context.scope.execution_context
    hits = state.get("hits", ())
    answer = state.get("answer")
    retrieved = decision is QueryDecision.RETRIEVE
    _log(
        runtime.context,
        query if not retrieved else state["raw_query"],
        state,
        retrieved=retrieved,
        hits=len(hits),
        answered=answer is not None if retrieved else None,
    )
    outcome = QueryOutcome(
        query=query,
        intent=state["intent"],
        confidence=state["confidence"],
        decision=decision,
        reason=state["reason"],
        guardrails=state["guardrails"],
        retrieval_performed=retrieved,
        hits=hits,
        message=(
            _retrieval_message(hits, answer, state)
            if retrieved
            else runtime.context.messages[decision]
        ),
        answer=answer,
        selected_sources=state.get("selected_sources", ()),
        context_resolution=state.get("context_resolution"),
        scope_degraded=context.scope_degraded,
        unavailable_documents=context.unavailable_documents,
        evidence=state.get("grade"),
        task=state.get("task"),
        plan=state.get("plan"),
        output_verdict=state.get("output_verdict"),
        generation_attempts=state.get("generation_attempts", 0),
    )
    return QueryGraphUpdate(outcome=outcome)


def _analysis_for_planning(state: QueryGraphState, query: str) -> QueryAnalysis:
    """The analysis to plan from, synthesized when no analyzer ran.

    A planner may be configured without an analyzer. Reaching this node at all
    means safety allowed the message and the intent warranted retrieval, so the
    synthesized verdict states exactly that rather than guessing at a task.
    """
    analysis = state.get("analysis")
    if analysis is not None:
        return analysis
    return QueryAnalysis(
        safety=SafetyVerdict.ALLOW,
        safety_categories=(),
        intent=state["intent"],
        task=state.get("task", QueryTask.LOOKUP),
        confidence=state["confidence"],
        reason_code="unanalyzed_document_question",
    )


def _search_queries(
    state: QueryGraphState,
    runtime: Runtime[QueryGraphRuntime],
) -> tuple[str, ...]:
    """The searches this attempt should run, always at least one.

    A rewrite (6C) supersedes the plan: the grader judged the evidence the plan
    produced and proposed different wording, so a corrective attempt runs that
    one search rather than repeating the whole decomposition.
    """
    current = state.get("effective_query", state["raw_query"])
    plan = state.get("plan")
    if plan is None or state.get("retrieval_attempts", 0) > 0:
        return (current,)
    return plan.search_queries[: runtime.context.max_subqueries] or (current,)


def _source_identity(hit: SearchHit) -> tuple[uuid.UUID, int, str]:
    """The same logical-source key the conversation ledger freezes per message.

    Chunk ids are not stable enough on their own: two attempts can return the
    same passage under different chunk rows, and deduplicating by chunk id would
    hand the model the same text twice under two citation numbers.
    """
    return hit.document_id, hit.index_generation, hit.logical_key


def _merge_hits(
    existing: tuple[SearchHit, ...],
    fresh: tuple[SearchHit, ...],
    *,
    limit: int | None,
) -> tuple[SearchHit, ...]:
    """Union both attempts by logical source, best-scoring first, within the limit.

    A passage found by both attempts keeps its better score and rank metadata,
    because the corrective wording is often the one that scores it properly.
    ``limit`` is the caller's source budget, so two attempts must not hand the
    model twice the passages one attempt would have; truncating after the sort
    keeps the strongest evidence from either attempt rather than the earliest.
    """
    best: dict[tuple[uuid.UUID, int, str], SearchHit] = {}
    for hit in (*existing, *fresh):
        identity = _source_identity(hit)
        incumbent = best.get(identity)
        if incumbent is None or hit.score > incumbent.score:
            best[identity] = hit
    merged = sorted(best.values(), key=lambda hit: hit.score, reverse=True)
    return tuple(merged if limit is None else merged[:limit])


def _should_retry(
    state: QueryGraphState,
    runtime: Runtime[QueryGraphRuntime],
    grade: EvidenceGrade,
) -> bool:
    """Decide the corrective branch here, where the trusted budget is visible.

    Conditional edges receive graph state only, and the attempt allowance is
    deliberately not in graph state - so the verdict is computed in the node
    that can see runtime context and read back as a plain boolean.
    """
    if grade.sufficient:
        return False
    if state.get("retrieval_attempts", 0) >= runtime.context.max_retrieval_attempts:
        return False
    suggested = grade.suggested_query
    # Nothing to correct with: an absent suggestion (the deterministic fallback
    # never offers one) or the same wording would only repeat the same search.
    current = state.get("effective_query", state["raw_query"])
    return bool(suggested) and suggested != current


def _evidence_insufficient(state: QueryGraphState) -> bool:
    """True only when a grader actually ran and judged the evidence too thin.

    An absent grade means ungraded - no grader configured, or grading was
    unavailable - and must never be read as a negative verdict.
    """
    grade = state.get("grade")
    return grade is not None and not grade.sufficient


def _retrieval_message(
    hits: tuple[SearchHit, ...],
    answer: GeneratedAnswer | None,
    state: QueryGraphState,
) -> str | None:
    """Policy text for a retrieval that produced no answer.

    Five silences look identical to a caller and mean entirely different things:
    nothing was found; the passages were found but graded as not supporting an
    answer; a draft was written and refused on safety grounds; a draft was
    written and could not be kept grounded; or the provider simply could not
    generate. Only the last is an outage, and reporting any of the others as one
    would send a user to check a status page over a working system.
    """
    if answer is not None:
        return None
    if not hits:
        return prompts.NO_SOURCES_MESSAGE
    verdict = state.get("output_verdict")
    if verdict is not None and not verdict.passed:
        return (
            prompts.UNSAFE_OUTPUT_MESSAGE
            if verdict.security_failure
            else prompts.REJECTED_ANSWER_MESSAGE
        )
    if _evidence_insufficient(state):
        return prompts.UNSUPPORTED_EVIDENCE_MESSAGE
    return prompts.GENERATION_UNAVAILABLE_MESSAGE


def _mentions_unavailable(query: str, context: QueryExecutionContext) -> bool:
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
    runtime: QueryGraphRuntime,
    query: str,
    state: QueryGraphState,
    *,
    retrieved: bool,
    hits: int,
    answered: bool | None,
) -> None:
    grade = state.get("grade")
    plan = state.get("plan")
    verdict = state.get("output_verdict")
    task = state.get("task")
    logger.info(
        "query",
        workspace_id=str(runtime.workspace_id),
        actor_id=str(runtime.actor.id),
        intent=state["intent"].value,
        decision=state["decision"].value,
        retrieval_performed=retrieved,
        hits=hits,
        reason=state["reason"],
        answered=answered,
        query_chars=len(query),
        task=task.value if task is not None else None,
        # None rather than 0/False on every non-retrieving path: grading was not
        # in play at all there, which is a different fact from "graded and found
        # wanting". Same convention `answered` already follows.
        retrieval_attempts=state.get("retrieval_attempts") if retrieved else None,
        planned_searches=len(plan.search_queries) if plan is not None else None,
        evidence_sufficient=grade.sufficient if grade is not None else None,
        evidence_conflict=grade.conflict_detected if grade is not None else None,
        generation_attempts=state.get("generation_attempts") if retrieved else None,
        output_issues=[issue.value for issue in verdict.issues] if verdict is not None else None,
    )
