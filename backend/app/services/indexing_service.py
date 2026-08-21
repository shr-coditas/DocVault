"""Lease, stage, and atomically activate a structured document index."""

import gzip
import hashlib
import json
import uuid
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from uuid6 import uuid7

from app.config import Settings, get_settings
from app.models.document_index import DocumentIndexRun, IndexRunStatus
from app.repository.document_chunk_repository import DocumentChunkRepository
from app.repository.document_index_repository import DocumentIndexRepository
from app.repository.document_repository import DocumentRepository
from app.services.ai_types import ExtractedArtifact, IndexOutcome, TextChunk
from app.services.audit_service import AuditService
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import Embedder
from app.services.storage_service import StorageService, index_artifact_key
from app.services.text_extraction_service import TextExtractionService

logger = structlog.stdlib.get_logger("docvault.indexing")

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]
_MAX_ERROR_CHARS = 2000


class OcrRequiredError(Exception):
    """A PDF has too little native text and OCR is intentionally not enabled."""


@dataclass(frozen=True, slots=True)
class _Claim:
    run_id: uuid.UUID
    lease_token: uuid.UUID
    document_id: uuid.UUID
    workspace_id: uuid.UUID
    storage_key: str
    file_name: str
    title: str


@dataclass(frozen=True, slots=True)
class _Prepared:
    artifact: ExtractedArtifact
    chunks: list[TextChunk]
    artifact_key: str


class IndexingService:
    normalizer_profile = "normalize-v2"

    def __init__(
        self,
        session_factory: SessionFactory,
        storage: StorageService,
        embedder: Embedder,
        extraction: TextExtractionService | None = None,
        chunking: ChunkingService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.storage = storage
        self.embedder = embedder
        self.settings = settings or get_settings()
        self.extraction = extraction or TextExtractionService()
        self.chunking = chunking or ChunkingService(self.settings, embedder)
        fingerprint = {
            "model": self.embedder.model_name,
            "dimensions": self.embedder.dimensions,
            "tokenizer": self.embedder.model_name,
            "query_prefix": self.settings.embedding_query_prefix,
            "normalization": "l2",
            "extractor": getattr(self.extraction, "profile_name", type(self.extraction).__name__),
            "normalizer": self.normalizer_profile,
            "chunker": self.chunking.profile_name,
        }
        digest = hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.embedding_profile = f"{self.embedder.model_name}:{digest}"[:255]

    def _lease_expiry(self) -> datetime:
        return datetime.now(UTC) + timedelta(minutes=self.settings.index_lease_minutes)

    async def claim(self, document_id: uuid.UUID) -> _Claim | None:
        async with self.session_factory() as session:
            documents = DocumentRepository(session)
            document = await documents.claim_for_indexing(document_id)
            if document is None:
                return None
            indexes = DocumentIndexRepository(session)
            if await indexes.competing_run(document_id) is not None:
                await session.rollback()
                return None

            lease_token = uuid7()
            run = await indexes.run_for_document(document_id)
            if run is None:
                run = DocumentIndexRun(
                    id=uuid7(),
                    workspace_id=document.workspace_id,
                    document_id=document.id,
                    extractor_profile="pending",
                    normalizer_profile=self.normalizer_profile,
                    chunker_profile=self.chunking.profile_name,
                    embedding_profile=self.embedding_profile,
                    status=IndexRunStatus.CLAIMED.value,
                    lease_token=lease_token,
                    lease_expires_at=self._lease_expiry(),
                    quality_metrics={},
                )
                indexes.add_run(run)
            else:
                await indexes.delete_staged(run.id)
                run.status = IndexRunStatus.CLAIMED.value
                run.lease_token = lease_token
                run.lease_expires_at = self._lease_expiry()
                run.error = None
                run.completed_at = None
                run.embedding_profile = self.embedding_profile
            await session.commit()
            return _Claim(
                run_id=run.id,
                lease_token=lease_token,
                document_id=document.id,
                workspace_id=document.workspace_id,
                storage_key=document.storage_key,
                file_name=document.file_name,
                title=document.title,
            )

    async def _transition(
        self,
        claim: _Claim,
        status: IndexRunStatus,
        *,
        artifact_key: str | None = None,
        artifact: ExtractedArtifact | None = None,
    ) -> None:
        async with self.session_factory() as session:
            run = await DocumentIndexRepository(session).get(claim.run_id)
            if run is None or run.lease_token != claim.lease_token:
                raise RuntimeError("index lease was lost")
            run.status = status.value
            run.lease_expires_at = self._lease_expiry()
            if artifact_key is not None:
                run.artifact_key = artifact_key
            if artifact is not None:
                run.extractor_profile = f"{artifact.parser_name}:{artifact.parser_version}"[:255]
                run.quality_metrics = artifact.quality_metrics
            await session.commit()

    async def load_and_prepare(self, claim: _Claim) -> _Prepared:
        await self._transition(claim, IndexRunStatus.EXTRACTING)
        data = await self.storage.read_all(claim.storage_key)
        artifact = await self.extraction.extract(data, claim.file_name)
        if "ocr_required" in artifact.warnings:
            raise OcrRequiredError("OCR required: insufficient native PDF text")
        if artifact.is_empty:
            raise RuntimeError("no extractable text")

        key = index_artifact_key(str(claim.workspace_id), str(claim.document_id))
        payload = gzip.compress(
            json.dumps(asdict(artifact), default=str, ensure_ascii=False).encode("utf-8")
        )
        await self.storage.put_bytes(key, payload, "application/gzip")
        await self._transition(claim, IndexRunStatus.CHUNKING, artifact_key=key, artifact=artifact)
        chunks = await self.chunking.split(artifact, document_title=claim.title)
        if not chunks:
            raise RuntimeError("no chunks produced")
        return _Prepared(artifact, chunks, key)

    @staticmethod
    def _batches[T](rows: Sequence[T], size: int) -> list[Sequence[T]]:
        return [rows[start : start + size] for start in range(0, len(rows), size)]

    async def stage(self, claim: _Claim, prepared: _Prepared) -> None:
        node_rows = [
            {
                "id": node.id,
                "run_id": claim.run_id,
                "workspace_id": claim.workspace_id,
                "document_id": claim.document_id,
                "parent_id": node.parent_id,
                "logical_path": node.logical_path,
                "ordinal": node.ordinal,
                "node_type": node.node_type,
                "heading_level": node.heading_level,
                "text": node.text,
                "source_spans": [asdict(location) for location in node.source_spans],
                "attributes": node.attributes,
                "confidence": node.confidence,
                "content_hash": hashlib.sha256((node.text or "").encode()).hexdigest(),
            }
            for node in prepared.artifact.nodes
        ]
        for batch in self._batches(node_rows, self.settings.index_node_batch_size):
            async with self.session_factory() as session:
                await DocumentIndexRepository(session).add_nodes(list(batch))
                await session.commit()
            await self._transition(claim, IndexRunStatus.CHUNKING)

        await self._transition(claim, IndexRunStatus.EMBEDDING)
        for chunk_batch in self._batches(
            prepared.chunks, self.settings.index_chunk_write_batch_size
        ):
            vectors: list[list[float]] = []
            for embedding_batch in self._batches(chunk_batch, self.settings.embedding_batch_size):
                vectors.extend(
                    await self.embedder.embed_texts(
                        [chunk.embedding_text for chunk in embedding_batch]
                    )
                )
            rows = []
            for chunk, vector in zip(chunk_batch, vectors, strict=True):
                if len(vector) != self.embedder.dimensions:
                    raise RuntimeError("embedding dimension mismatch while staging")
                if chunk.structural_node_id is None:
                    raise RuntimeError("chunk has no structural node")
                rows.append(
                    {
                        "id": uuid7(),
                        "run_id": claim.run_id,
                        "document_id": claim.document_id,
                        "workspace_id": claim.workspace_id,
                        "logical_key": chunk.logical_key,
                        "chunk_index": chunk.chunk_index,
                        "structural_node_id": chunk.structural_node_id,
                        "parent_node_id": chunk.parent_node_id,
                        "ordinal_in_parent": chunk.ordinal_in_parent,
                        "chunk_type": chunk.chunk_type,
                        "heading_path": list(chunk.heading_path),
                        "breadcrumb": chunk.breadcrumb,
                        "content": chunk.content,
                        "embedding_text": chunk.embedding_text,
                        "lexical_text": chunk.lexical_text,
                        "embedding_token_count": chunk.token_count,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "source_spans": [asdict(span) for span in chunk.source_spans],
                        "language": chunk.language,
                        "content_hash": chunk.content_hash,
                        "metadata": chunk.metadata,
                        "embedding_profile_id": self.embedding_profile,
                        "embedding_model": self.embedder.model_name,
                        "embedding": vector,
                    }
                )
            async with self.session_factory() as session:
                await DocumentChunkRepository(session).add_many(rows)
                await session.commit()
            await self._transition(claim, IndexRunStatus.EMBEDDING)
        await self._transition(claim, IndexRunStatus.STAGED)

    async def activate(self, claim: _Claim, prepared: _Prepared) -> IndexOutcome:
        async with self.session_factory() as session:
            indexes = DocumentIndexRepository(session)
            chunks = DocumentChunkRepository(session)
            run = await indexes.get(claim.run_id)
            document = await DocumentRepository(session).get(claim.document_id)
            if run is None or run.lease_token != claim.lease_token:
                raise RuntimeError("index lease was lost before activation")
            if document is None or document.deleted_at is not None:
                run.status = IndexRunStatus.FAILED.value
                run.error = "document went away"
                run.completed_at = datetime.now(UTC)
                await session.commit()
                return IndexOutcome(claim.document_id, "skipped", detail="document went away")
            if await indexes.node_count(claim.run_id) != len(prepared.artifact.nodes):
                raise RuntimeError("staged node count mismatch")
            if await indexes.invalid_node_count(claim.run_id):
                raise RuntimeError("staged node hash validation failed")
            if await chunks.count_for_run(claim.run_id) != len(prepared.chunks):
                raise RuntimeError("staged chunk count mismatch")
            if await chunks.invalid_for_run(
                claim.run_id,
                embedding_profile=self.embedding_profile,
                embedding_model=self.embedder.model_name,
                dimensions=self.embedder.dimensions,
            ):
                raise RuntimeError("staged chunk profile, vector, or hash validation failed")

            run.status = IndexRunStatus.ACTIVE.value
            run.completed_at = datetime.now(UTC)
            run.lease_expires_at = self._lease_expiry()
            document.indexed = True
            document.indexed_at = datetime.now(UTC)
            document.index_error = None
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
        return IndexOutcome(claim.document_id, "indexed", chunk_count=len(prepared.chunks))

    async def record_failure(
        self, document_id: uuid.UUID, detail: str, claim: _Claim | None = None
    ) -> IndexOutcome:
        message = detail[:_MAX_ERROR_CHARS]
        async with self.session_factory() as session:
            document = await DocumentRepository(session).get(document_id)
            if claim is not None:
                indexes = DocumentIndexRepository(session)
                run = await indexes.get(claim.run_id)
                if run is not None and run.lease_token == claim.lease_token:
                    await indexes.delete_staged(run.id)
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
            return IndexOutcome(document_id, "skipped", detail="already claimed or indexed")
        try:
            prepared = await self.load_and_prepare(claim)
            await self.stage(claim, prepared)
            return await self.activate(claim, prepared)
        except Exception as exc:
            logger.warning("index_failed", document_id=str(document_id), error=type(exc).__name__)
            detail = (
                str(exc)
                if isinstance(exc, RuntimeError | OcrRequiredError)
                else f"{type(exc).__name__}: {exc}"
            )
            return await self.record_failure(document_id, detail, claim)
