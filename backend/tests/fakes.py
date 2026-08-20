"""Deterministic test doubles for the document-intelligence seams.

Every double here is offline and reproducible. That is the point: the
authorization tests must be able to assert exactly which chunks reached the
model, and a real embedding provider would make those assertions approximate.
"""

import json
import math
import zlib
from collections.abc import Sequence
from typing import Any

from app.ai.agent.prompt_utils import Supervision
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

    def __init__(
        self,
        reply: str = "A grounded answer [1].",
        replies: Sequence[str] | None = None,
    ) -> None:
        # ``replies`` scripts a sequence for the tests that let a draft be
        # rejected and rewritten; the last one repeats if it runs out.
        self.replies = list(replies) if replies else [reply]
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
            text=self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)],
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


class RecordingSupervisor:
    """Base for the supervisor doubles: remembers the brief, decides nothing.

    ``ainvoke`` is the whole interface, because the real supervisor is a chat
    model pinned to the `Supervision` schema and nothing more. Recording the
    human message proves what was actually sent - several tests assert that a
    hostile question never reached a model, or that the history did.
    """

    def __init__(self) -> None:
        self.briefs: list[str] = []

    @property
    def histories(self) -> list[list[dict[str, str]]]:
        """The conversation each decision was shown, oldest brief first."""
        return [json.loads(brief)["history"] for brief in self.briefs]

    @property
    def questions(self) -> list[str]:
        return [json.loads(brief)["question"] for brief in self.briefs]

    async def ainvoke(self, messages: Sequence[Any], **kwargs: Any) -> Supervision:
        brief = str(messages[-1].content)
        self.briefs.append(brief)
        return self.decide(json.loads(brief))

    def decide(self, brief: dict[str, Any]) -> Supervision:
        raise NotImplementedError


class FakeSupervisor(RecordingSupervisor):
    """Scripted, one decision per pass, and loud if a test scripts too few."""

    def __init__(self, *decisions: Supervision) -> None:
        super().__init__()
        self.decisions = list(decisions)

    def decide(self, brief: dict[str, Any]) -> Supervision:
        if not self.decisions:
            raise AssertionError("the supervisor was asked one more time than scripted")
        return self.decisions.pop(0)


class UnavailableSupervisor(RecordingSupervisor):
    def decide(self, brief: dict[str, Any]) -> Supervision:
        raise RuntimeError("the supervisor provider is unavailable")


class OfflineSupervisor(RecordingSupervisor):
    """A supervisor with no model behind it: search once, then answer.

    It reads the brief exactly as the real one does, so the integration suite
    drives the whole loop - guard, classify, supervise, retrieve, write -
    without a provider and without a scripted answer per test.
    """

    def __init__(
        self,
        *,
        rewrite_to: str | None = None,
        clarify: bool = False,
        unsupported: bool = False,
    ) -> None:
        super().__init__()
        # The three things a test usually wants to pin: how a follow-up gets
        # resolved, that it cannot be, and that the sources do not answer it.
        self.rewrite_to = rewrite_to
        self.clarify = clarify
        self.unsupported = unsupported

    def decide(self, brief: dict[str, Any]) -> Supervision:
        if self.clarify:
            return Supervision(action="clarify", reason="ambiguous_referent")
        if brief["sources"]:
            if self.unsupported:
                return Supervision(action="unsupported", reason="does_not_answer_it")
            return Supervision(action="answer", reason="sources_in_hand")
        if brief["searches_left"]:
            question = self.rewrite_to or brief["question"]
            return Supervision(
                action="search",
                question=question if self.rewrite_to else "",
                searches=[question],
                reason="nothing_retrieved_yet",
            )
        return Supervision(action="unsupported", reason="nothing_found")
