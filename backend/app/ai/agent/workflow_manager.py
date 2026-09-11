"""Build the scope model and run one query turn."""

from typing import Any

from langchain_core.runnables import Runnable

from app.ai.agent.agent_manager import Context
from app.ai.agent.graph_manager import get_graph
from app.ai.agent.llm_response_dto import ScopeDecision
from app.config import Settings
from app.services.ai_types import QueryExecutionContext, QueryOutcome


def create_decision_model(settings: Settings) -> Runnable[Any, Any]:
    from langchain.chat_models import init_chat_model

    provider = settings.agent_model_provider or settings.llm_provider
    model_name = settings.agent_model or settings.llm_model
    credentials: dict[str, Any] = {}
    if provider == "google_genai" and settings.google_studio_api_key:
        credentials["api_key"] = settings.google_studio_api_key
    model = init_chat_model(
        model=model_name,
        model_provider=provider,
        temperature=0,
        max_tokens=settings.agent_max_output_tokens,
        **credentials,
    )
    return model.with_structured_output(ScopeDecision)


async def run_workflow(
    user_question: str,
    execution: QueryExecutionContext,
    context: Context,
) -> QueryOutcome:
    """Run one bounded query turn and return the completed outcome."""
    state = await get_graph().ainvoke(
        {
            "question": user_question,
            "history": execution.history,
            "document_summaries": execution.document_summaries,
            "summaries_complete": execution.summaries_complete,
        },
        context=context,
        config={"recursion_limit": context.settings.agent_max_steps},
    )
    outcome: QueryOutcome = state["outcome"]
    return outcome
