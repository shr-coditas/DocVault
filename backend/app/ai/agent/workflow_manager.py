"""Running one turn: the model the supervisor uses, and the entry point."""

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import Runnable

from app.ai.agent.agent_manager import Context
from app.ai.agent.graph_manager import get_graph
from app.ai.agent.llm_response_dto import Supervision
from app.config import Settings
from app.services.ai_types import ConversationTurn, QueryOutcome


def create_decision_model(settings: Settings) -> Runnable[Any, Any]:
    from langchain.chat_models import init_chat_model

    provider = settings.llm_provider
    credentials: dict[str, Any] = {}
    if provider == "google_genai" and settings.google_studio_api_key:
        credentials["api_key"] = settings.google_studio_api_key
    model = init_chat_model(
        model=settings.llm_model,
        model_provider=provider,
        temperature=0,
        max_tokens=settings.agent_max_output_tokens,
        **credentials,
    )
    return model.with_structured_output(Supervision)


def serialize_conversation(history: Sequence[ConversationTurn]) -> list[AnyMessage]:
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


async def run_workflow(
    user_question: str,
    past_conversation_history: Sequence[ConversationTurn],
    context: Context,
) -> QueryOutcome:
    """Run one query turn and return what to tell the caller.

    ``recursion_limit`` is a backstop, not the bound. The real limits are the
    search and draft budgets the supervisor is clamped to; if this limit is ever
    the thing that stops a turn, something upstream is wrong.
    """
    state = await get_graph().ainvoke(
        {
            "messages": [
                *serialize_conversation(past_conversation_history),
                HumanMessage(content=user_question),
            ]
        },
        context=context,
        config={"recursion_limit": context.settings.agent_max_steps},
    )
    outcome: QueryOutcome = state["outcome"]
    return outcome
