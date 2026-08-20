"""Define and compile the transient behavioral-parity graph once."""

from functools import lru_cache
from typing import cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.ai.query_graph import nodes, routes
from app.ai.query_graph.runtime import QueryGraphRuntime
from app.ai.query_graph.state import (
    QueryGraphInput,
    QueryGraphOutput,
    QueryGraphState,
)
from app.services.ai_types import QueryOutcome

CompiledQueryGraph = CompiledStateGraph[
    QueryGraphState,
    QueryGraphRuntime,
    QueryGraphInput,
    QueryGraphOutput,
]


def build_query_graph() -> CompiledQueryGraph:
    graph = StateGraph(
        QueryGraphState, # state schema
        context_schema=QueryGraphRuntime, 
        input_schema=QueryGraphInput,
        output_schema=QueryGraphOutput,
    )
    graph.add_node("guard", nodes.guard_input) 
    graph.add_node(routes.ANALYZE, nodes.analyze_query)
    graph.add_node(routes.LOAD_CONTEXT, nodes.load_context)
    graph.add_node(routes.RESOLVE_CONTEXT, nodes.resolve_context)
    graph.add_node(routes.PLAN, nodes.plan_retrieval)
    graph.add_node(routes.RETRIEVE, nodes.retrieve)
    graph.add_node(routes.GRADE, nodes.grade_evidence)
    graph.add_node(routes.REWRITE, nodes.rewrite_query)
    graph.add_node(routes.GENERATE, nodes.generate)
    graph.add_node(routes.VALIDATE, nodes.validate_output)
    graph.add_node(routes.FINALIZE, nodes.finalize)

    graph.add_edge(START, "guard")
    #conditional
    graph.add_conditional_edges(
        "guard", #source node
        routes.after_guard, # function that determines the next node based on the current state
        {routes.ANALYZE: routes.ANALYZE, routes.FINALIZE: routes.FINALIZE}, # mapping of possible next nodes based on the decision made in the after_guard function
    )
    graph.add_conditional_edges(
        routes.ANALYZE,
        routes.after_analysis,
        {
            routes.LOAD_CONTEXT: routes.LOAD_CONTEXT,
            routes.FINALIZE: routes.FINALIZE,
        },
    )
    graph.add_conditional_edges(
        routes.LOAD_CONTEXT,
        routes.after_context,
        {
            routes.RESOLVE_CONTEXT: routes.RESOLVE_CONTEXT,
            routes.FINALIZE: routes.FINALIZE,
        },
    )
    graph.add_conditional_edges(
        routes.RESOLVE_CONTEXT,
        routes.after_resolution,
        {routes.PLAN: routes.PLAN, routes.FINALIZE: routes.FINALIZE},
    )
    graph.add_conditional_edges(
        routes.PLAN,
        routes.after_plan,
        {routes.RETRIEVE: routes.RETRIEVE, routes.FINALIZE: routes.FINALIZE},
    )
    graph.add_edge(routes.RETRIEVE, routes.GRADE)
    graph.add_conditional_edges(
        routes.GRADE,
        routes.after_grade,
        {routes.REWRITE: routes.REWRITE, routes.GENERATE: routes.GENERATE},
    )
    # Cycle one, corrective retrieval. `grade` refuses to set `retry_retrieval`
    # once the runtime-owned attempt budget is spent, so this returns to
    # `retrieve` a bounded number of times; `recursion_limit` is the backstop,
    # not the bound.
    graph.add_edge(routes.REWRITE, routes.RETRIEVE)
    graph.add_edge(routes.GENERATE, routes.VALIDATE)
    # Cycle two, the single regeneration, bounded the same way by `validate`.
    # These are the only two cycles in the graph, and neither edge decides its
    # own budget - both read a boolean a node computed against runtime context.
    graph.add_conditional_edges(
        routes.VALIDATE,
        routes.after_validation,
        {routes.GENERATE: routes.GENERATE, routes.FINALIZE: routes.FINALIZE},
    )
    graph.add_edge(routes.FINALIZE, END)
    return graph.compile(name="docvault_query")


@lru_cache(maxsize=1)
def get_query_graph() -> CompiledQueryGraph:
    return build_query_graph()


async def run_query_graph(
    raw_query: str,
    runtime: QueryGraphRuntime,
    *,
    recursion_limit: int,
) -> QueryOutcome:
    result = await get_query_graph().ainvoke(
        QueryGraphInput(raw_query=raw_query),
        config={"recursion_limit": recursion_limit},
        context=runtime,
    )
    return cast(QueryGraphOutput, result)["outcome"]
