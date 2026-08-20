"""Build the graph: the six nodes, and the edges between them.

    START
      |
      v
    guard ------------------------------+   a message that is empty, oversized
      |                                 |   or full of zero-width characters
      v                                 |
    classify --------------------------->   a greeting, a non-document request,
      |                                 |   a known attack, or a dead scope
      v                                 |
    supervise <--------+--------+       |   the one model call
      |     |          |        |       |
      |     +-> retrieve        |       |   run its searches, report back
      |     |                   |       |
      |     +-> write ----------+       |   draft, check, and either finish
      |            |                    |   or hand a rejection back
      v            v                    v
    respond <--------------------------- ---> END

Nodes that pick a branch write ``next_step``; ``route`` reads it back. The
mapping beside each conditional edge is the list of places that node is allowed
to send a turn, so the whole shape of the agent is the twelve lines below.
"""

from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.ai.agent.agent_manager import (
    AgentContext,
    State,
    classify,
    guard,
    respond,
    retrieve,
    supervise,
    write,
)


def route(state: State) -> str:
    """Where the node that just ran decided to go."""
    return state["next_step"]


def build_graph() -> Any:
    builder = StateGraph(State, context_schema=AgentContext)

    builder.add_node("guard", guard)
    builder.add_node("classify", classify)
    builder.add_node("supervise", supervise)
    builder.add_node("retrieve", retrieve)
    builder.add_node("write", write)
    builder.add_node("respond", respond)

    builder.add_edge(START, "guard")
    builder.add_conditional_edges("guard", route, ["classify", "respond"])
    builder.add_conditional_edges("classify", route, ["supervise", "respond"])
    builder.add_conditional_edges("supervise", route, ["retrieve", "write", "respond"])
    # The two loops back to the supervisor. Neither is bounded here: `retrieve`
    # and `write` are bounded by the budgets `agent_manager` clamps the
    # supervisor to, and the recursion limit is only a backstop behind those.
    builder.add_edge("retrieve", "supervise")
    builder.add_conditional_edges("write", route, ["supervise", "respond"])
    builder.add_edge("respond", END)

    return builder.compile(name="docvault_query")


@lru_cache(maxsize=1)
def get_graph() -> Any:
    """Compiled once per process; everything per-request is in the context."""
    return build_graph()
