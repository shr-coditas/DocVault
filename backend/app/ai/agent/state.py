"""What one query turn accumulates as it moves through the graph."""

from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from app.services.ai_types import (
    AnswerDraft,
    GeneratedAnswer,
    GuardrailOutcome,
    OutputVerdict,
    QueryDecision,
    QueryIntent,
    QueryOutcome,
    SearchHit,
)


class QueryInput(TypedDict):
    """What a caller starts a turn with: the chat, ending in the new question."""

    messages: Annotated[list[AnyMessage], add_messages]


class QueryOutput(TypedDict):
    outcome: QueryOutcome


class QueryState(QueryInput):
    """The turn's working memory.

    ``messages`` is the conversation itself - the earlier turns of this chat,
    then the question being asked now. History lives here rather than behind a
    node that fetches it, because history is state: the supervisor reads it on
    every pass to keep a follow-up resolved, and a node that loaded it once
    would only have to put it in a field like this one anyway.

    Nothing here decides what the asker may read. Actor, workspace and document
    scope sit in ``AgentContext``, out of reach of anything a model writes.
    """

    # The question as the supervisor resolved it - the same text as the last
    # message until a follow-up gets rewritten against the history.
    question: str
    guardrails: GuardrailOutcome
    intent: QueryIntent
    confidence: float
    decision: QueryDecision
    reason: str
    resolved_from_history: NotRequired[bool]

    # Retrieval. Sources accumulate across searches rather than replacing each
    # other: a second search is an addition to the evidence, not a correction of
    # it, and every hit on every pass went through the same permission filter.
    sources: NotRequired[tuple[SearchHit, ...]]
    # What the supervisor asked for on this pass, and how many passes have run.
    # The counter is rounds, not queries: one decision to search costs one,
    # whether it named one wording or three.
    searches: NotRequired[tuple[str, ...]]
    searches_run: NotRequired[int]
    # Every wording already tried, so a second pass cannot re-run the first.
    searched: NotRequired[tuple[str, ...]]

    # Generation. ``draft`` is model text that has not been checked yet, and it
    # is cleared the moment it fails a check so that no later node, log or
    # ledger row can reach text that was refused.
    draft: NotRequired[AnswerDraft | None]
    supplied: NotRequired[tuple[SearchHit, ...]]
    drafts_written: NotRequired[int]
    verdict: NotRequired[OutputVerdict | None]
    answer: NotRequired[GeneratedAnswer | None]

    # Set when the supervisor ends a retrieving turn without an answer, which is
    # a different fact from the answerer having failed.
    unsupported: NotRequired[bool]

    outcome: NotRequired[QueryOutcome]
