"""Local BGE embeddings through FastEmbed."""

import asyncio
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from app.config import Settings, get_settings
from app.models.document_chunk import EMBEDDING_DIMENSIONS


class FastEmbedEmbedder:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_name = self.settings.embedding_model_name

        # imported here, not at module scope, so merely importing this module
        # does not load ONNX Runtime - tests that override the seam never pay it
        from fastembed import TextEmbedding

        # threads = how many threads ONNX Runtime may use for inference
        # cache_dir = where the downloaded model files are kept
        self._model: Any = TextEmbedding(
            model_name=self.model_name,
            threads=self.settings.embedding_threads,
            cache_dir=self.settings.fastembed_cache_dir,
        )

        # Probe rather than trusting a metadata table: it is version-proof, and
        # it warms the model so the first real request is not the slow one.
        # (embed a throwaway string purely to see how many dimensions come back)
        probe = self._embed_sync(["dimension probe"])
        self.dimensions = len(probe[0])
        if self.dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"{self.model_name} produces {self.dimensions}-d vectors but "
                f"document_chunks.embedding expects {EMBEDDING_DIMENSIONS}; "
                "document_chunks.embedding would reject these"
            )

    def _embed_sync(self, texts: Sequence[str]) -> list[list[float]]:
        """used to embed texts synchronously in a thread, to avoid blocking the event loop"""
        vectors = self._model.embed(list(texts), batch_size=self.settings.embedding_batch_size)
        return [vector.tolist() for vector in vectors]

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages. Offloaded - this is the heaviest CPU step in indexing."""
        if not texts:
            return []
        # asyncio.to_thread runs _embed_sync on a worker thread so the event loop
        # stays free. The event loop is the single thread that drives every
        # coroutine in the process: it interleaves tasks by switching at `await`
        # points. A long synchronous call like ONNX inference never yields, so
        # running it inline would stall every other request until it finished.
        return await asyncio.to_thread(self._embed_sync, texts)

    async def embed_query(self, text: str) -> list[float]:
        """Embed one query, with the retrieval prefix bge expects."""
        prefixed = f"{self.settings.embedding_query_prefix}{text}"
        vectors = await asyncio.to_thread(self._embed_sync, [prefixed])
        return vectors[0]

    def count_tokens(self, text: str) -> int:
        """Use the exact tokenizer owned by the configured embedding model."""
        return int(self._model.token_count(text))


# lru_cache memoises on the arguments; with none, the first call computes and
# every later call returns that same object. maxsize=1 makes the intent explicit
# - there is only ever one entry to hold.
@lru_cache(maxsize=1)
def get_default_embedder() -> FastEmbedEmbedder:
    """Process-wide singleton.

    Loading the ONNX model takes seconds, so it must happen once per process and
    never per request. Both the API (via ``dependencies.get_embedder``) and the
    indexing script go through here.
    """
    return FastEmbedEmbedder()
