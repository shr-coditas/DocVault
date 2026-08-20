"""Run one question through the query agent, printing every node as it runs.

Built for a debugger. Put a breakpoint in any node in ``agent_manager`` and
launch this with the "Script: current file" configuration; the graph runs in
this process, so the call stack you land in is the real one.

The selected user is intentional, exactly as in ``evaluate_retrieval``:
retrieval must go through the same permission predicates the API uses, so
debugging a turn cannot accidentally show you passages the user cannot read.

    # the real supervisor, the real corpus - needs a provider key
    uv run python -m app.scripts.run_agent "What is the carry-over limit?" \
        --workspace-id <uuid> --user-id <uuid>

    # the same graph and the same corpus, but no provider key and no billing:
    # search once, then answer over whatever came back
    uv run python -m app.scripts.run_agent "..." --workspace-id <uuid> \
        --user-id <uuid> --offline

    # a follow-up, to watch the supervisor resolve it against the history
    uv run python -m app.scripts.run_agent "What about contractors?" \
        --workspace-id <uuid> --user-id <uuid> \
        --history "What is the carry-over limit?" "Five days [1]."
"""

import argparse
import asyncio
import json
import uuid
from typing import Any, cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from app.ai.agent.agent_manager import Context, State
from app.ai.agent.graph_manager import get_graph
from app.ai.agent.prompt_utils import Supervision
from app.ai.agent.workflow_manager import as_messages, build_supervisor
from app.config import get_settings
from app.db.session import async_session_factory, dispose_engine
from app.repository.user_repository import UserRepository
from app.services.ai_types import ConversationTurn, SearchMode
from app.services.answer_service import AnswerService
from app.services.embedding_service import get_default_embedder
from app.services.llm_service import get_default_chat_model
from app.services.reranking_service import get_default_reranker
from app.services.search_service import SearchService
from app.utils.logging import configure_logging


class OfflineSupervisor:
    """Search once, then answer. The loop, with no provider and no billing.

    It reads the brief the same way the real supervisor does, so ``--offline``
    still exercises every node, both budgets and the whole state contract - the
    only thing it does not exercise is the prompt.
    """

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> Supervision:
        brief = json.loads(str(messages[-1].content))
        if brief["sources"]:
            return Supervision(action="answer", reason="sources_in_hand")
        if brief["searches_left"]:
            return Supervision(
                action="search",
                searches=[brief["question"]],
                reason="nothing_retrieved_yet",
            )
        return Supervision(action="unsupported", reason="nothing_found")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one turn through the query agent.")
    parser.add_argument("question")
    parser.add_argument("--workspace-id", type=uuid.UUID, required=True)
    parser.add_argument("--user-id", type=uuid.UUID, required=True)
    parser.add_argument(
        "--history",
        nargs="*",
        default=[],
        metavar="MESSAGE",
        help="alternating user and assistant messages, oldest first",
    )
    parser.add_argument("--document-id", type=uuid.UUID, action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--mode", type=SearchMode, choices=list(SearchMode), default=SearchMode.HYBRID
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the scripted supervisor instead of calling a provider",
    )
    return parser.parse_args()


def _history(messages: list[str]) -> tuple[ConversationTurn, ...]:
    if len(messages) % 2:
        raise ValueError(
            "--history takes alternating user and assistant messages, so an even number"
        )
    return tuple(
        ConversationTurn(messages[index], messages[index + 1])
        for index in range(0, len(messages), 2)
    )


def _describe(node: str, update: dict[str, Any]) -> str:
    """One readable line per node, without dumping whole search hits."""
    interesting = []
    for key in ("decision", "reason", "next_step", "question", "searches"):
        if key in update:
            interesting.append(f"{key}={update[key]!r}")
    if "sources" in update:
        interesting.append(f"sources={len(update['sources'])}")
    if "answer" in update and update["answer"] is not None:
        interesting.append("answer=<generated>")
    if "verdict" in update and update["verdict"] is not None:
        interesting.append(f"issues={[issue.value for issue in update['verdict'].issues]}")
    return f"{node:<10} {' '.join(interesting)}"


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings)
    history = _history(args.history)

    async with async_session_factory() as session:
        actor = await UserRepository(session).get(args.user_id)
        if actor is None:
            raise ValueError("that user does not exist")

        context = Context(
            actor=actor,
            workspace_id=args.workspace_id,
            search=SearchService(session, get_default_embedder(), reranker=get_default_reranker()),
            answers=AnswerService(get_default_chat_model()),
            # Duck-typed on purpose: the graph only ever calls `ainvoke`.
            supervisor=cast(
                Runnable[Any, Any],
                OfflineSupervisor() if args.offline else build_supervisor(settings),
            ),
            settings=settings,
            document_ids=tuple(args.document_id) if args.document_id else None,
            limit=args.limit,
            mode=args.mode,
        )

        # `astream` rather than `ainvoke`: the same run, but every node's update
        # arrives as it is produced, which is what makes a loop legible.
        final: State | None = None
        async for step in get_graph().astream(
            {"messages": [*as_messages(history), HumanMessage(content=args.question)]},
            context=context,
            config={"recursion_limit": settings.agent_max_steps},
            stream_mode="updates",
        ):
            for node, update in step.items():
                print(_describe(node, update))
                if node == "respond":
                    final = update

    print()
    if final is None:
        print("the graph ended without responding, which should not happen")
        return 1

    outcome = final["outcome"]
    print(f"decision   {outcome.decision.value}  ({outcome.reason})")
    print(f"question   {outcome.query}")
    print(f"hits       {len(outcome.hits)}  supplied {len(outcome.selected_sources)}")
    if outcome.answer is not None:
        markers = [citation.marker for citation in outcome.answer.citations]
        print(f"citations  {markers}")
        print()
        print(outcome.answer.text)
    else:
        print()
        print(outcome.message)
    return 0


async def _main() -> int:
    try:
        return await _run(_arguments())
    finally:
        await dispose_engine()


def main() -> None:
    raise SystemExit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
