"""Read, chunk, embed, and atomically replace one document's search index."""

import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from uuid6 import uuid7

from app.config import Settings, get_settings
from app.models.document_index import DocumentIndexRun, IndexRunStatus
from app.repository.document_chunk_repository import DocumentChunkRepository
from app.repository.document_index_repository import DocumentIndexRepository
from app.repository.document_repository import DocumentRepository
from app.services.ai_types import IndexOutcome, TextChunk
from app.services.audit_service import AuditService
from app.services.chunking_service import ChunkingService, build_embedding_text
from app.services.embedding_service import FastEmbedEmbedder
from app.services.storage_service import StorageService
from app.services.text_extraction_service import TextExtractionService

logger = structlog.stdlib.get_logger("docvault.indexing")

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]
_MAX_ERROR_CHARS = 2000


@dataclass(frozen=True, slots=True)
class _Claim:
    run_id: uuid.UUID
    document_id: uuid.UUID
    workspace_id: uuid.UUID
    storage_key: str
    file_name: str
    title: str


@dataclass(frozen=True, slots=True)
class _Prepared:
    chunks: list[TextChunk]
    vectors: list[list[float]]


class IndexingService:
    def __init__(
        self,
        session_factory: SessionFactory,
        storage: StorageService,
        embedder: FastEmbedEmbedder,
        settings: Settings | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.storage = storage
        self.embedder = embedder
        self.settings = settings or get_settings()
        self.extraction = TextExtractionService()
        self.chunking = ChunkingService(self.settings, embedder)

    async def claim(self, document_id: uuid.UUID) -> _Claim | None:
        async with self.session_factory() as session:
            document = await DocumentRepository(session).claim_for_indexing(document_id)
            if document is None:
                return None
            run = DocumentIndexRun(
                id=uuid7(),
                document_id=document.id,
                status=IndexRunStatus.PROCESSING.value,
            )
            DocumentIndexRepository(session).add(run)
            document.index_error = None
            await session.commit()
            return _Claim(
                run_id=run.id,
                document_id=document.id,
                workspace_id=document.workspace_id,
                storage_key=document.storage_key,
                file_name=document.file_name,
                title=document.title,
            )

    async def prepare(self, claim: _Claim) -> _Prepared:
        data = await self.storage.read_all(claim.storage_key)
        extracted = await self.extraction.extract(data, claim.file_name)
        if extracted.is_empty:
            raise RuntimeError("no extractable text")
        chunks = await self.chunking.split(extracted, document_title=claim.title)
        if not chunks:
            raise RuntimeError("no chunks produced")
        embedding_texts = [build_embedding_text(claim.title, chunk) for chunk in chunks]
        vectors = await self.embedder.embed_texts(embedding_texts)
        if len(vectors) != len(chunks):
            raise RuntimeError("embedding count does not match chunk count")
        if any(len(vector) != self.embedder.dimensions for vector in vectors):
            raise RuntimeError("embedding dimension mismatch")
        return _Prepared(chunks, vectors)

    async def activate(self, claim: _Claim, prepared: _Prepared) -> IndexOutcome:
        async with self.session_factory() as session:
            run = await DocumentIndexRepository(session).get(claim.run_id)
            document = await DocumentRepository(session).get(claim.document_id)
            if run is None:
                raise RuntimeError("index run disappeared")
            if document is None or document.deleted_at is not None:
                run.status = IndexRunStatus.FAILED.value
                run.error = "document went away"
                run.completed_at = datetime.now(UTC)
                await session.commit()
                return IndexOutcome(claim.document_id, "skipped", detail=run.error)

            chunks = DocumentChunkRepository(session)
            await chunks.delete_for_document(claim.document_id)
            await chunks.add_many(
                [
                    {
                        "id": uuid7(),
                        "document_id": claim.document_id,
                        "workspace_id": claim.workspace_id,
                        "chunk_index": chunk.chunk_index,
                        "chunk_type": chunk.chunk_type,
                        "section_path": chunk.section_path,
                        "content": chunk.content,
                        "embedding": vector,
                    }
                    for chunk, vector in zip(prepared.chunks, prepared.vectors, strict=True)
                ]
            )
            now = datetime.now(UTC)
            document.indexed = True
            document.indexed_at = now
            document.index_error = None
            run.status = IndexRunStatus.COMPLETED.value
            run.error = None
            run.completed_at = now
            AuditService(session).record(
                action="document.indexed",
                resource_type="document",
                resource_id=claim.document_id,
                workspace_id=claim.workspace_id,
                actor_id=None,
                chunk_count=len(prepared.chunks),
                embedding_model=self.embedder.model_name,
            )
            await session.commit()
        return IndexOutcome(
            claim.document_id,
            "indexed",
            chunk_count=len(prepared.chunks),
        )

    async def record_failure(
        self,
        document_id: uuid.UUID,
        detail: str,
        claim: _Claim | None = None,
    ) -> IndexOutcome:
        message = detail[:_MAX_ERROR_CHARS]
        async with self.session_factory() as session:
            document = await DocumentRepository(session).get(document_id)
            if claim is not None:
                run = await DocumentIndexRepository(session).get(claim.run_id)
                if run is not None:
                    run.status = IndexRunStatus.FAILED.value
                    run.error = message
                    run.completed_at = datetime.now(UTC)
            if document is not None:
                document.index_error = message
                AuditService(session).record(
                    action="document.index_failed",
                    resource_type="document",
                    resource_id=document_id,
                    workspace_id=document.workspace_id,
                    actor_id=None,
                    reason=message,
                )
            await session.commit()
        return IndexOutcome(document_id, "failed", detail=message)

    async def index_document(self, document_id: uuid.UUID) -> IndexOutcome:
        claim = await self.claim(document_id)
        if claim is None:
            return IndexOutcome(document_id, "skipped", detail="already indexed or unavailable")
        try:
            prepared = await self.prepare(claim)
            return await self.activate(claim, prepared)
        except Exception as exc:
            logger.warning(
                "index_failed",
                document_id=str(document_id),
                error=type(exc).__name__,
            )
            detail = str(exc) if isinstance(exc, RuntimeError) else f"{type(exc).__name__}: {exc}"
            return await self.record_failure(document_id, detail, claim)
