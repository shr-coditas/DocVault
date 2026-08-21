from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.ai.agent.agent_manager import (
    Context,
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
    builder = StateGraph(State, context_schema=Context)
    builder.add_node("guard", guard)
    builder.add_node("classify", classify)
    builder.add_node("supervise", supervise)
    builder.add_node("retrieve", retrieve)
    builder.add_node("write", write)
    builder.add_node("respond", respond)
    builder.add_edge(START, "guard")
    builder.add_conditional_edges(
        "guard", route, ["classify", "respond"]
    ) 
    builder.add_conditional_edges("classify", route, ["supervise", "respond"])
    builder.add_conditional_edges("supervise", route, ["retrieve", "write", "respond"])
    builder.add_edge("retrieve", "supervise")
    builder.add_conditional_edges("write", route, ["supervise", "respond"])
    builder.add_edge("respond", END)
    return builder.compile(name="docvault_query")


@lru_cache(maxsize=1)
def get_graph() -> Any:
    return build_graph()


def draw_graph() -> str:
    """
        uv run python -c "from app.ai.agent.graph_manager import draw_graph; print(draw_graph())"
    """
    return str(get_graph().get_graph().draw_mermaid())
