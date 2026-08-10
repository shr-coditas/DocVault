"""Offline cross-encoder reranking behind a testable protocol."""

import asyncio
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Protocol

from app.config import Settings, get_settings


class Reranker(Protocol):
    model_name: str

    async def rerank(self, query: str, passages: Sequence[str]) -> list[float]: ...


class FastEmbedReranker:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_name = self.settings.reranker_model_name
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(
                model_name=self.model_name,
                cache_dir=self.settings.fastembed_cache_dir,
            )
        return self._model

    def _rerank_sync(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        rows = self._load().rerank(query, list(passages))
        scores = []
        for row in rows:
            value = getattr(row, "score", row)
            scores.append(float(value))
        return scores

    async def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        return await asyncio.to_thread(self._rerank_sync, query, passages)


class IdentityReranker:
    """Deterministic fallback/test double that preserves candidate order."""

    model_name = "identity"

    async def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        return [float(len(passages) - index) for index in range(len(passages))]


@lru_cache(maxsize=1)
def get_default_reranker() -> FastEmbedReranker:
    return FastEmbedReranker()
