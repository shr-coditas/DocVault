"""Deterministic test doubles for the document-intelligence seams.

Every double here is offline and reproducible. That is the point: the
authorization tests must be able to assert exactly which chunks reached the
model, and a real embedding provider would make those assertions approximate.
"""

import math
import zlib
from collections.abc import Sequence

from app.services.ai_types import (
    ContextReason,
    ContextResolution,
    ConversationTurn,
)
from app.services.contextual_query_service import ResolverUnavailableError
from app.services.llm_service import Completion, LLMUnavailableError
from app.services.reranking_service import IdentityReranker

DIMENSIONS = 384


class FakeReranker(IdentityReranker):
    pass


def _tokens(text: str) -> list[str]:
    return [token.strip(".,;:!?()[]\"'").lower() for token in text.split()]


class FakeEmbedder:
    """Token-hash embedder: same dimensionality as bge-small, no model.

    Each token lands in one of 384 buckets by CRC32 - ``hash()`` would not do,
    because Python randomises it per process and the vectors have to be stable
    across runs. Two texts sharing a rare word therefore land close together,
    which is what lets a test say "this document is retrievable by this query"
    and mean it.
    """

    model_name = "fake-token-hash"
    dimensions = DIMENSIONS

    def __init__(self) -> None:
        # every batch this embedder was asked to embed, for assertions
        self.embedded_batches: list[list[str]] = []
        # every query embedded. This is how the intent gate is proved: embedding
        # is a strict precondition of the vector query, so an empty list after a
        # request means nothing reached the index - a stronger claim than
        # "no hits came back".
        self.embedded_queries: list[str] = []

    def vector(self, text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for token in _tokens(text):
            if token:
                vector[zlib.crc32(token.encode()) % DIMENSIONS] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # an empty or punctuation-only chunk still needs a unit vector, or
            # cosine distance against it is undefined
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded_batches.append(list(texts))
        return [self.vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        # no bge prefix here on purpose: it would add tokens that pollute the
        # buckets and weaken the similarity the tests rely on
        self.embedded_queries.append(text)
        return self.vector(text)

    def count_tokens(self, text: str) -> int:
        return max(1, len(_tokens(text)))


class ExplodingEmbedder:
    """Fails on demand, to exercise the indexing failure path."""

    model_name = "exploding"
    dimensions = DIMENSIONS

    def __init__(self, message: str = "embedding provider unavailable") -> None:
        self.message = message

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError(self.message)

    async def embed_query(self, text: str) -> list[float]:
        raise RuntimeError(self.message)

    def count_tokens(self, text: str) -> int:
        return max(1, len(_tokens(text)))


class FakeChatModel:
    """Scripted generation that records exactly what it was asked.

    ``prompts`` is the load-bearing part. Asserting on the response only proves
    what a user *saw*; asserting on what this received proves what actually left
    the building - which is the claim the access-control tests need to make, and
    the one an inaccessible document must never satisfy.
    """

    model_name = "fake-chat"

    def __init__(self, reply: str = "A grounded answer [1].") -> None:
        self.reply = reply
        # (system, user) for every call, in order
        self.prompts: list[tuple[str, str]] = []

    @property
    def calls(self) -> int:
        return len(self.prompts)

    @property
    def last_user_prompt(self) -> str:
        return self.prompts[-1][1]

    async def complete(self, system: str, user: str) -> Completion:
        self.prompts.append((system, user))
        return Completion(
            text=self.reply,
            model=self.model_name,
            input_tokens=11,
            output_tokens=7,
        )


class UnavailableChatModel:
    """Every call fails, to exercise the degrade-to-sources path.

    Raises the same ``LLMUnavailableError`` the real seam collapses provider
    failures into, because the point of that single exception type is that the
    caller cannot tell a timeout from a rate limit - and neither can this.
    """

    model_name = "unavailable"

    def __init__(self, message: str = "the model provider is unavailable") -> None:
        self.message = message
        self.calls = 0

    async def complete(self, system: str, user: str) -> Completion:
        self.calls += 1
        raise LLMUnavailableError(self.message)


class FakeContextualResolver:
    """Scripted follow-up resolver with call history independent of generation."""

    def __init__(
        self,
        *,
        standalone_query: str | None = None,
        needs_clarification: bool = False,
        reason_code: ContextReason = ContextReason.REWRITTEN,
    ) -> None:
        self.standalone_query = standalone_query
        self.needs_clarification = needs_clarification
        self.reason_code = reason_code
        self.calls: list[tuple[str, tuple[ConversationTurn, ...]]] = []

    async def resolve(
        self,
        current_message: str,
        history: Sequence[ConversationTurn],
    ) -> ContextResolution:
        frozen_history = tuple(history)
        self.calls.append((current_message, frozen_history))
        return ContextResolution(
            standalone_query=self.standalone_query or current_message,
            used_history=bool(history),
            needs_clarification=self.needs_clarification,
            reason_code=self.reason_code,
        )


class UnavailableContextualResolver:
    def __init__(self) -> None:
        self.calls = 0

    async def resolve(
        self,
        current_message: str,
        history: Sequence[ConversationTurn],
    ) -> ContextResolution:
        self.calls += 1
        raise ResolverUnavailableError("resolver unavailable")
