"""Index uploaded documents into pgvector. Run by hand.

In the container (the supported way - the image bakes the ONNX model in, and uv
is a build-stage tool that is deliberately absent from the runtime image, so
`python` here is already the venv's):

    docker compose run --rm api python -m app.scripts.index_documents

On the host, if the venv can load every extension module it needs:

    uv --directory backend run python -m app.scripts.index_documents
    uv --directory backend run python -m app.scripts.index_documents --limit 10
    uv --directory backend run python -m app.scripts.index_documents --workspace-id <uuid>
    uv --directory backend run python -m app.scripts.index_documents --retry-failed

Picks up documents where ``indexed = false`` and indexes them, oldest first.
Re-running is safe: an indexed document is never selected again, and re-indexing
one replaces its chunks rather than adding to them.

Sequential on purpose. Concurrency is what the orchestrated workflow will add;
building a half-version of it here would be work thrown away.

Exits 0 when nothing failed, 1 otherwise, so it can be used from CI or cron
without parsing the output.
"""

import argparse
import asyncio
import uuid

import structlog

from app.config import get_settings
from app.db.session import async_session_factory, dispose_engine
from app.repository.document_repository import DocumentRepository
from app.services.embedding_service import get_default_embedder
from app.services.indexing_service import IndexingService
from app.services.storage_service import StorageService
from app.utils.logging import configure_logging

logger = structlog.stdlib.get_logger("docvault.indexing")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="index_documents",
        description="Index not-yet-indexed documents into pgvector.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="maximum documents to process (default: index_batch_limit)",
    )
    parser.add_argument(
        "--workspace-id",
        type=uuid.UUID,
        default=None,
        help="restrict to one workspace",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="also retry documents that have already failed max_index_attempts times",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings)

    limit = args.limit if args.limit is not None else settings.index_batch_limit
    max_attempts = None if args.retry_failed else settings.max_index_attempts

    async with async_session_factory() as session:
        pending = await DocumentRepository(session).unindexed_ids(
            limit=limit,
            workspace_id=args.workspace_id,
            max_attempts=max_attempts,
        )

    if not pending:
        logger.info("nothing_to_index")
        return 0

    logger.info("indexing_started", pending=len(pending), limit=limit)

    # loading the ONNX model takes seconds; do it once, before the loop, and
    # only once we know there is work to do
    service = IndexingService(
        session_factory=async_session_factory,
        storage=StorageService(settings),
        embedder=get_default_embedder(),
        settings=settings,
    )

    counts = {"indexed": 0, "skipped": 0, "failed": 0}
    for position, document_id in enumerate(pending, start=1):
        outcome = await service.index_document(document_id)
        counts[outcome.status] += 1
        logger.info(
            "document_processed",
            progress=f"{position}/{len(pending)}",
            document_id=str(document_id),
            status=outcome.status,
            chunk_count=outcome.chunk_count,
            # `detail` is a reason or an exception type, never document content
            detail=outcome.detail,
        )

    logger.info("indexing_finished", **counts)
    return 1 if counts["failed"] else 0


async def _main(args: argparse.Namespace) -> int:
    try:
        return await _run(args)
    finally:
        # The module-level engine is created on import, so it needs disposing even
        # when the run raised - but it MUST happen on this loop. Disposing from a
        # second asyncio.run() tries to close asyncpg sockets bound to a loop that
        # the first run already closed, which surfaces as "Event loop is closed"
        # (posix) or "'NoneType' object has no attribute 'send'" (Windows proactor).
        await dispose_engine()


def main() -> None:
    raise SystemExit(asyncio.run(_main(_parse_args())))


if __name__ == "__main__":
    main()
