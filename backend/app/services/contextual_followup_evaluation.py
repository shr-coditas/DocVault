"""Labeled, provider-optional evaluation for contextual query resolution."""

import json
import math
import statistics
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.ai_types import (
    ContextReason,
    ContextResolution,
    ConversationTurn,
    QueryIntent,
)
from app.services.contextual_query_service import (
    ContextualQueryResolver,
    ResolverUnavailableError,
)
from app.services.guardrail_service import GuardrailService
from app.services.intent_service import IntentClassifier, RuleBasedIntentClassifier


@dataclass(frozen=True, slots=True)
class FollowupEvaluationCase:
    id: str
    category: str
    current_message: str
    history: tuple[ConversationTurn, ...]
    expected_standalone_queries: tuple[str, ...]
    expected_needs_clarification: bool
    should_call_resolver: bool
    forbidden_terms: tuple[str, ...] = ()


def _required_string(item: dict[str, Any], field: str, case_id: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"case {case_id!r} needs a non-empty {field}")
    return value


def load_followup_cases(path: Path) -> list[FollowupEvaluationCase]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("follow-up evaluation file must contain a non-empty JSON list")

    cases: list[FollowupEvaluationCase] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"case at index {index} must be an object")
        case_id = _required_string(item, "id", f"index-{index}")
        if case_id in seen_ids:
            raise ValueError(f"duplicate case id {case_id!r}")
        seen_ids.add(case_id)

        raw_history = item.get("history", [])
        if not isinstance(raw_history, list):
            raise ValueError(f"case {case_id!r} history must be a list")
        history: list[ConversationTurn] = []
        for turn in raw_history:
            if not isinstance(turn, dict):
                raise ValueError(f"case {case_id!r} has an invalid history turn")
            history.append(
                ConversationTurn(
                    user_message=_required_string(turn, "user", case_id),
                    assistant_message=_required_string(turn, "assistant", case_id),
                )
            )

        expected = item.get("expected_standalone_queries")
        if (
            not isinstance(expected, list)
            or not expected
            or not all(isinstance(value, str) and value.strip() for value in expected)
        ):
            raise ValueError(f"case {case_id!r} needs non-empty expected_standalone_queries")
        clarification = item.get("expected_needs_clarification")
        should_call = item.get("should_call_resolver")
        forbidden = item.get("forbidden_terms", [])
        if not isinstance(clarification, bool) or not isinstance(should_call, bool):
            raise ValueError(f"case {case_id!r} needs boolean expectations")
        if not isinstance(forbidden, list) or not all(
            isinstance(value, str) and value.strip() for value in forbidden
        ):
            raise ValueError(f"case {case_id!r} forbidden_terms must be strings")

        cases.append(
            FollowupEvaluationCase(
                id=case_id,
                category=_required_string(item, "category", case_id),
                current_message=_required_string(item, "current_message", case_id),
                history=tuple(history),
                expected_standalone_queries=tuple(expected),
                expected_needs_clarification=clarification,
                should_call_resolver=should_call,
                forbidden_terms=tuple(forbidden),
            )
        )
    return cases


def _normalized(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    if normalized.endswith(("?", ".", "!")):
        normalized = normalized[:-1]
    return " ".join(normalized.split())


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


async def evaluate_followups(
    cases: list[FollowupEvaluationCase],
    resolver: ContextualQueryResolver,
    *,
    guardrails: GuardrailService | None = None,
    intent_classifier: IntentClassifier | None = None,
) -> dict[str, object]:
    """Evaluate the same raw-query gates that precede the resolver in production."""
    guardrail_service = guardrails or GuardrailService()
    classifier = intent_classifier or RuleBasedIntentClassifier()
    details: list[dict[str, object]] = []
    latencies: list[float] = []
    resolver_calls = 0
    provider_failures = 0
    exact_matches = 0
    scored_exact_cases = 0
    clarification_matches = 0
    expected_clarifications = 0
    clarification_true_positives = 0
    non_ambiguous_cases = 0
    unnecessary_clarifications = 0
    routing_matches = 0
    widening_failures = 0

    for case in cases:
        started = time.perf_counter()
        guardrail_outcome = await guardrail_service.run(case.current_message)
        intent: QueryIntent | None = None
        eligible = False
        if guardrail_outcome.passed:
            intent = (await classifier.classify(case.current_message)).intent
            eligible = intent is QueryIntent.DOCUMENT_QUESTION and bool(case.history)

        resolver_called = False
        provider_failed = False
        if eligible:
            resolver_called = True
            resolver_calls += 1
            try:
                resolution = await resolver.resolve(case.current_message, case.history)
            except ResolverUnavailableError:
                provider_failed = True
                provider_failures += 1
                resolution = ContextResolution(
                    standalone_query=case.current_message,
                    used_history=False,
                    needs_clarification=False,
                    reason_code=ContextReason.FALLBACK,
                )
        else:
            resolution = ContextResolution(
                standalone_query=case.current_message,
                used_history=False,
                needs_clarification=False,
                reason_code=ContextReason.UNCHANGED,
            )

        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        expected_queries = {_normalized(value) for value in case.expected_standalone_queries}
        exact_match = _normalized(resolution.standalone_query) in expected_queries
        if not provider_failed and not case.expected_needs_clarification:
            scored_exact_cases += 1
            exact_matches += int(exact_match)
        clarification_match = resolution.needs_clarification == case.expected_needs_clarification
        clarification_matches += int(clarification_match)
        if case.expected_needs_clarification:
            expected_clarifications += 1
            clarification_true_positives += int(resolution.needs_clarification)
        else:
            non_ambiguous_cases += 1
            unnecessary_clarifications += int(resolution.needs_clarification)
        routing_match = resolver_called == case.should_call_resolver
        routing_matches += int(routing_match)
        widened = any(
            _normalized(term) in _normalized(resolution.standalone_query)
            for term in case.forbidden_terms
        )
        widening_failures += int(widened)
        details.append(
            {
                "id": case.id,
                "category": case.category,
                "intent": intent.value if intent is not None else None,
                "guardrails_passed": guardrail_outcome.passed,
                "resolver_called": resolver_called,
                "provider_failed": provider_failed,
                "actual_standalone_query": resolution.standalone_query,
                "reason_code": resolution.reason_code.value,
                "needs_clarification": resolution.needs_clarification,
                "exact_match": (
                    exact_match
                    if not provider_failed and not case.expected_needs_clarification
                    else None
                ),
                "clarification_match": clarification_match,
                "routing_match": routing_match,
                "scope_widening_failure": widened,
                "latency_ms": latency_ms,
            }
        )

    total = len(cases) or 1
    return {
        "cases": len(cases),
        "metrics": {
            "standalone_exact_accuracy": (
                exact_matches / scored_exact_cases if scored_exact_cases else 0.0
            ),
            "standalone_exact_scored_cases": scored_exact_cases,
            "clarification_accuracy": clarification_matches / total,
            "ambiguity_recall": (
                clarification_true_positives / expected_clarifications
                if expected_clarifications
                else 0.0
            ),
            "unnecessary_clarification_rate": (
                unnecessary_clarifications / non_ambiguous_cases if non_ambiguous_cases else 0.0
            ),
            "resolver_routing_accuracy": routing_matches / total,
            "resolver_call_rate": resolver_calls / total,
            "provider_failures": provider_failures,
            "provider_failure_rate_per_resolver_call": (
                provider_failures / resolver_calls if resolver_calls else 0.0
            ),
            "scope_widening_failure_rate": widening_failures / total,
            "latency_ms": {
                "median": statistics.median(latencies) if latencies else 0.0,
                "p95": _percentile(latencies, 0.95),
            },
        },
        "results": details,
    }
