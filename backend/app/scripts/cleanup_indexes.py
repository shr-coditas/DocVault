"""Fail expired indexing leases and remove retained extraction artifacts."""

import asyncio
from datetime import UTC, datetime, timedelta

import structlog

from app.config import get_settings
from app.db.session import async_session_factory, dispose_engine
from app.models.document_index import IndexRunStatus
from app.repository.document_index_repository import DocumentIndexRepository
from app.services.storage_service import StorageService
from app.utils.logging import configure_logging

logger = structlog.stdlib.get_logger("docvault.index_cleanup")


async def _run() -> int:
    settings = get_settings()
    configure_logging(settings)
    now = datetime.now(UTC)
    async with async_session_factory() as session:
        repository = DocumentIndexRepository(session)
        expired = await repository.expired_nonterminal(now)
        for run in expired:
            await repository.delete_staged(run.id)
            run.status = IndexRunStatus.FAILED.value
            run.error = "index lease expired"
            run.completed_at = now
        await session.commit()

    cutoff = now - timedelta(days=settings.index_artifact_retention_days)
    storage = StorageService(settings)
    removed = 0
    async with async_session_factory() as session:
        runs = await DocumentIndexRepository(session).artifacts_before(cutoff)
        for run in runs:
            if run.artifact_key is None:
                continue
            await storage.delete(run.artifact_key)
            run.artifact_key = None
            removed += 1
        await session.commit()
    logger.info("index_cleanup", expired=len(expired), artifacts_removed=removed)
    return 0


async def _main() -> int:
    try:
        return await _run()
    finally:
        await dispose_engine()


def main() -> None:
    raise SystemExit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
