"""The query graph: deterministic screening, then a supervised loop.

    START --> screen --+--> supervise --+--> retrieve --+
                       |         ^      |               |
                       |         +------+--> write -----+
                       |         |      |          |
                       |         +------+----------+
                       |                |
                       +--> respond <---+--> END

              screen    deterministic; ends the turn or wakes the supervisor
              supervise the only model call, and the only router
              retrieve  runs the searches it asked for, reports back
              write     drafts and checks; a clean or unsafe draft is final,
                        a merely ungrounded one goes back for a decision
              respond   assembles the outcome

``screen`` is the deterministic gate. ``supervise`` is the only node that asks a
model anything, and it is the only node that decides where the turn goes next.
``retrieve`` and ``write`` do one job each and report back to it.

There are two static edges here because everything else is decided inside the
nodes and returned as a ``Command``. That is the trade: the picture above is not
in the wiring, it is in the return types of the five functions in ``nodes.py`` -
and in exchange there is exactly one place to look to find out why a turn did
what it did.
"""

from collections.abc import Sequence
from functools import lru_cache
from typing import cast

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.ai.agent import nodes
from app.ai.agent.context import AgentContext
from app.ai.agent.state import QueryInput, QueryOutput, QueryState
from app.services.ai_types import ConversationTurn, QueryOutcome

CompiledQueryGraph = CompiledStateGraph[QueryState, AgentContext, QueryInput, QueryOutput]


def build_graph() -> CompiledQueryGraph:
    builder = StateGraph(
        QueryState,
        context_schema=AgentContext,
        input_schema=QueryInput,
        output_schema=QueryOutput,
    )
    builder.add_node("screen", nodes.screen)
    builder.add_node("supervise", nodes.supervise)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("write", nodes.write)
    builder.add_node("respond", nodes.respond)

    builder.add_edge(START, "screen")
    builder.add_edge("respond", END)
    return builder.compile(name="docvault_query")


@lru_cache(maxsize=1)
def get_graph() -> CompiledQueryGraph:
    """Compiled once per process; the per-request parts all live in the context."""
    return build_graph()


def as_messages(history: Sequence[ConversationTurn]) -> list[AnyMessage]:
    """The conversation as the graph carries it, oldest first.

    The caller has already narrowed this to turns whose sources this actor can
    still open, so nothing here re-exposes a document that was revoked between
    then and now.
    """
    messages: list[AnyMessage] = []
    for turn in history:
        messages.append(HumanMessage(content=turn.user_message))
        messages.append(AIMessage(content=turn.assistant_message))
    return messages


async def run_query_graph(
    question: str,
    history: Sequence[ConversationTurn],
    context: AgentContext,
) -> QueryOutcome:
    """Run one turn and return what to tell the caller.

    ``recursion_limit`` is a backstop, not the bound. The real limits are the
    search and draft budgets the supervisor is clamped to in ``nodes``; if this
    limit is ever the thing that stops a turn, something upstream is wrong.
    """
    result = await get_graph().ainvoke(
        QueryInput(messages=[*as_messages(history), HumanMessage(content=question)]),
        context=context,
        config={"recursion_limit": context.settings.agent_max_steps},
    )
    return cast(QueryOutput, result)["outcome"]
