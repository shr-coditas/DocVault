"""Permission-filtered semantic, lexical, and hybrid retrieval."""

import uuid
from dataclasses import dataclass
from typing import cast

import structlog
from sqlalchemy import Row
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.user import User
from app.repository.document_chunk_repository import DocumentChunkRepository
from app.services.ai_types import (
    ChunkType,
    ScoreBreakdown,
    SearchHit,
    SearchMode,
    SearchResult,
)
from app.services.document_access import document_access_filter
from app.services.embedding_service import FastEmbedEmbedder
from app.services.reranking_service import FastEmbedReranker

logger = structlog.stdlib.get_logger("docvault.search")
MAX_LIMIT = 50
MAX_DOCUMENT_IDS = 10


@dataclass(slots=True)
class _Candidate:
    chunk: DocumentChunk
    document: Document
    semantic: float | None = None
    lexical: float | None = None
    fusion: float | None = None
    rerank: float | None = None

    @property
    def final_score(self) -> float:
        for score in (self.rerank, self.fusion, self.semantic, self.lexical):
            if score is not None:
                return score
        return 0.0


class SearchService:
    def __init__(
        self,
        session: AsyncSession,
        embedder: FastEmbedEmbedder,
        settings: Settings | None = None,
        reranker: FastEmbedReranker | None = None,
    ) -> None:
        self.session = session
        self.embedder = embedder
        self.chunks = DocumentChunkRepository(session)
        self.settings = settings or get_settings()
        self.reranker = reranker

    async def search(
        self,
        actor: User,
        workspace_id: uuid.UUID,
        query: str,
        *,
        mode: SearchMode = SearchMode.HYBRID,
        limit: int | None = None,
        semantic_min_score: float | None = None,
        document_id: uuid.UUID | None = None,
        document_ids: list[uuid.UUID] | None = None,
    ) -> SearchResult:
        effective_limit = min(max(1, limit or self.settings.search_default_limit), MAX_LIMIT)
        if document_id is not None and document_ids:
            raise ValueError("document_id and document_ids are mutually exclusive")
        scope = [document_id] if document_id is not None else list(document_ids or [])
        if len(scope) > MAX_DOCUMENT_IDS:
            raise ValueError(f"at most {MAX_DOCUMENT_IDS} document ids are allowed")
        effective_semantic_min_score = (
            self.settings.search_semantic_min_score
            if semantic_min_score is None
            else semantic_min_score
        )
        if not query.strip():
            return SearchResult(query, effective_limit, effective_semantic_min_score, (), mode)

        access = await document_access_filter(self.session, actor.id, workspace_id)
        semantic_rows: list[Row[tuple[DocumentChunk, Document, float]]] = []
        lexical_rows: list[Row[tuple[DocumentChunk, Document, float]]] = []
        candidate_limit = max(effective_limit, self.settings.search_candidate_limit)
        if mode in {SearchMode.SEMANTIC, SearchMode.HYBRID}:
            query_vector = await self.embedder.embed_query(query)
            semantic_rows = await self.chunks.search_semantic(
                workspace_id=workspace_id,
                query_vector=query_vector,
                limit=candidate_limit,
                semantic_min_score=effective_semantic_min_score,
                access=access,
                document_ids=scope or None,
                iterative_scan=self.settings.search_iterative_scan,
            )
        if mode in {SearchMode.LEXICAL, SearchMode.HYBRID}:
            lexical_rows = await self.chunks.search_lexical(
                workspace_id=workspace_id,
                query=query,
                limit=candidate_limit,
                access=access,
                document_ids=scope or None,
            )

        candidates = self._fuse(mode, semantic_rows, lexical_rows)
        if mode is SearchMode.HYBRID and self.reranker is not None:
            candidates = await self._rerank(query, candidates)
        candidates = self._diversify(candidates, single_document=len(scope) == 1)
        selected = candidates[:effective_limit]
        hits = tuple(self._to_hit(candidate, mode) for candidate in selected)

        logger.info(
            "search",
            workspace_id=str(workspace_id),
            actor_id=str(actor.id),
            mode=mode.value,
            hits=len(hits),
            limit=effective_limit,
            semantic_min_score=effective_semantic_min_score,
            unfiltered=access is None,
        )
        return SearchResult(query, effective_limit, effective_semantic_min_score, hits, mode)

    async def semantic(
        self,
        actor: User,
        workspace_id: uuid.UUID,
        query: str,
        *,
        limit: int | None = None,
        semantic_min_score: float | None = None,
        document_id: uuid.UUID | None = None,
    ) -> SearchResult:
        """Backward-compatible explicit semantic entrypoint."""
        return await self.search(
            actor,
            workspace_id,
            query,
            mode=SearchMode.SEMANTIC,
            limit=limit,
            semantic_min_score=semantic_min_score,
            document_id=document_id,
        )

    def _fuse(
        self,
        mode: SearchMode,
        semantic_rows: list[Row[tuple[DocumentChunk, Document, float]]],
        lexical_rows: list[Row[tuple[DocumentChunk, Document, float]]],
    ) -> list[_Candidate]:
        by_id: dict[uuid.UUID, _Candidate] = {}
        for rank, (chunk, document, score) in enumerate(semantic_rows, start=1):
            candidate = by_id.setdefault(chunk.id, _Candidate(chunk, document))
            candidate.semantic = float(score)
            if mode is SearchMode.HYBRID:
                candidate.fusion = (candidate.fusion or 0.0) + 1 / (
                    self.settings.search_rrf_k + rank
                )
        for rank, (chunk, document, score) in enumerate(lexical_rows, start=1):
            candidate = by_id.setdefault(chunk.id, _Candidate(chunk, document))
            candidate.lexical = float(score)
            if mode is SearchMode.HYBRID:
                candidate.fusion = (candidate.fusion or 0.0) + 1 / (
                    self.settings.search_rrf_k + rank
                )
        return sorted(by_id.values(), key=lambda row: row.final_score, reverse=True)[
            : self.settings.search_fusion_limit
        ]

    async def _rerank(self, query: str, candidates: list[_Candidate]) -> list[_Candidate]:
        head = candidates[: self.settings.search_rerank_limit]
        tail = candidates[self.settings.search_rerank_limit :]
        passages = [
            "\n".join(part for part in (row.chunk.section_path, row.chunk.content) if part)
            for row in head
        ]
        scores = await self.reranker.rerank(query, passages) if self.reranker else []
        if len(scores) != len(head):
            raise RuntimeError("reranker returned a score count that does not match candidates")
        for row, score in zip(head, scores, strict=True):
            row.rerank = score
        return [*sorted(head, key=lambda row: row.final_score, reverse=True), *tail]

    def _diversify(
        self, candidates: list[_Candidate], *, single_document: bool
    ) -> list[_Candidate]:
        if single_document:
            return candidates
        counts: dict[uuid.UUID, int] = {}
        selected = []
        for candidate in candidates:
            count = counts.get(candidate.document.id, 0)
            if count >= self.settings.search_per_document_limit:
                continue
            counts[candidate.document.id] = count + 1
            selected.append(candidate)
        return selected

    @staticmethod
    def _to_hit(candidate: _Candidate, mode: SearchMode) -> SearchHit:
        chunk = candidate.chunk
        document = candidate.document
        return SearchHit(
            document_id=document.id,
            document_title=document.title,
            file_name=document.file_name,
            chunk_id=chunk.id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            score=candidate.final_score,
            mode=mode,
            chunk_type=cast(ChunkType, chunk.chunk_type),
            section_path=chunk.section_path,
            scores=ScoreBreakdown(
                semantic=candidate.semantic,
                lexical=candidate.lexical,
                fusion=candidate.fusion,
                rerank=candidate.rerank,
            ),
        )
