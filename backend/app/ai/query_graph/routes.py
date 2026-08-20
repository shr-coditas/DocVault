"""Conditional-edge decisions for the bounded query graph."""

from typing import Literal

from app.ai.query_graph.state import QueryGraphState
from app.services.ai_types import QueryDecision

ANALYZE = "analyze"
LOAD_CONTEXT = "load_context"
RESOLVE_CONTEXT = "resolve_context"
PLAN = "plan"
RETRIEVE = "retrieve"
GRADE = "grade"
REWRITE = "rewrite"
GENERATE = "generate"
VALIDATE = "validate"
FINALIZE = "finalize"

AfterGuardRoute = Literal["analyze", "finalize"]
AfterAnalysisRoute = Literal["load_context", "finalize"]
AfterContextRoute = Literal["resolve_context", "finalize"]
AfterResolutionRoute = Literal["plan", "finalize"]
AfterPlanRoute = Literal["retrieve", "finalize"]
AfterGradeRoute = Literal["rewrite", "generate"]
AfterValidationRoute = Literal["generate", "finalize"]

# Every decision that ends the turn without retrieving. Listed once here because
# three separate edges ask the same question, and an intent added later must not
# have to be remembered in three places.
_TERMINAL_BEFORE_RETRIEVAL = frozenset(
    {
        QueryDecision.BLOCK,
        QueryDecision.CLARIFY,
        QueryDecision.SCOPE_UNAVAILABLE,
    }
)


def after_guard(state: QueryGraphState) -> AfterGuardRoute:
    return "finalize" if state.get("decision") is QueryDecision.BLOCK else "analyze"


def after_analysis(state: QueryGraphState) -> AfterAnalysisRoute:
    return "load_context" if state["decision"] is QueryDecision.RETRIEVE else "finalize"


def after_context(state: QueryGraphState) -> AfterContextRoute:
    return (
        "finalize"
        if state.get("decision") is QueryDecision.SCOPE_UNAVAILABLE
        else "resolve_context"
    )


def after_resolution(state: QueryGraphState) -> AfterResolutionRoute:
    return "finalize" if state.get("decision") in _TERMINAL_BEFORE_RETRIEVAL else "plan"


def after_plan(state: QueryGraphState) -> AfterPlanRoute:
    """A planner may ask the user a question instead of searching (6D).

    The plan node has already converted that into ``CLARIFY``, so this reads the
    decision rather than the plan - keeping one definition of "this turn ends
    without retrieving" across all four pre-retrieval edges.
    """
    return "finalize" if state.get("decision") in _TERMINAL_BEFORE_RETRIEVAL else "retrieve"


def after_grade(state: QueryGraphState) -> AfterGradeRoute:
    """The corrective-retrieval cycle, and the grade node already bounded it.

    ``retry_retrieval`` is false whenever the evidence was judged sufficient,
    the attempt budget is spent, grading was unavailable, or no usable rewrite
    was offered - so this edge cannot loop on its own.
    """
    return "rewrite" if state.get("retry_retrieval") else "generate"


def after_validation(state: QueryGraphState) -> AfterValidationRoute:
    """The regeneration cycle (6E), bounded the same way for the same reason.

    ``regenerate`` is only ever set for a quality failure with attempts left. A
    security failure is terminal by construction, so no rejected draft can talk
    its way into another call.
    """
    return "generate" if state.get("regenerate") else "finalize"
