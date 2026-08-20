"""Deterministic test doubles for the document-intelligence seams.

Every double here is offline and reproducible. That is the point: the
authorization tests must be able to assert exactly which chunks reached the
model, and a real embedding provider would make those assertions approximate.
"""

import math
import zlib
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

from app.services.ai_types import (
    ContextReason,
    ContextResolution,
    ConversationTurn,
    EvidenceGrade,
    OutputVerdict,
    QueryAnalysis,
    QueryPlan,
    QueryTask,
    SearchHit,
)
from app.services.contextual_query_service import ResolverUnavailableError
from app.services.llm_service import Completion, LLMUnavailableError
from app.services.query_analysis_service import (
    LayeredQueryAnalyzer,
    StructuredQueryAnalyzer,
)
from app.services.reranking_service import IdentityReranker
from app.services.structured_model_service import StructuredModelUnavailableError

DIMENSIONS = 384
StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


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


class FakeStructuredModel:
    """Validated scripted outputs with exact prompt/schema call history."""

    model_name = "fake-structured"

    def __init__(self, *responses: BaseModel | dict[str, object]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, type[BaseModel], float]] = []

    async def complete(
        self,
        system: str,
        user: str,
        schema: type[StructuredOutput],
        *,
        timeout_seconds: float,
    ) -> StructuredOutput:
        self.calls.append((system, user, schema, timeout_seconds))
        if not self.responses:
            raise AssertionError("no fake structured response remains")
        response = self.responses.pop(0)
        return schema.model_validate(response)


class UnavailableStructuredModel:
    model_name = "unavailable-structured"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        system: str,
        user: str,
        schema: type[StructuredOutput],
        *,
        timeout_seconds: float,
    ) -> StructuredOutput:
        self.calls += 1
        raise StructuredModelUnavailableError("structured model unavailable")


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


class FakeEvidenceGrader:
    """Scripted sufficiency verdicts, one per graded attempt.

    ``calls`` records the query and how many passages were judged, which is what
    the corrective-retrieval tests assert on: that the second grading saw the
    rewritten wording and the merged evidence, not the original pair.
    """

    def __init__(self, *grades: EvidenceGrade) -> None:
        self.grades = list(grades)
        self.calls: list[tuple[str, int]] = []

    async def grade(
        self,
        query: str,
        hits: Sequence[SearchHit],
        plan: QueryPlan,
    ) -> EvidenceGrade:
        self.calls.append((query, len(hits)))
        if not self.grades:
            raise AssertionError("no fake evidence grade remains")
        return self.grades.pop(0)


class UnavailableEvidenceGrader:
    """Every grading fails, to prove an ungraded query still gets its answer."""

    def __init__(self) -> None:
        self.calls = 0

    async def grade(
        self,
        query: str,
        hits: Sequence[SearchHit],
        plan: QueryPlan,
    ) -> EvidenceGrade:
        self.calls += 1
        raise StructuredModelUnavailableError("evidence grader unavailable")


def offline_query_analyzer() -> LayeredQueryAnalyzer:
    """A fully offline analyzer for the HTTP+DB graph-path tests.

    Layered over an always-unavailable structured model, so every message is
    decided by the same rule classifier the real analyzer already falls back to.
    That keeps the database parity tests deterministic and network-free - the
    ``FakeChatModel`` of query analysis - without asserting a specific scripted
    verdict, since those tests care about orchestration, not classification.
    """
    return LayeredQueryAnalyzer(StructuredQueryAnalyzer(UnavailableStructuredModel()))


class FakeQueryAnalyzer:
    """Scripted safety/intent/task verdicts, with the messages it judged."""

    def __init__(self, *analyses: QueryAnalysis) -> None:
        self.analyses = list(analyses)
        self.calls: list[str] = []

    async def analyze(self, query: str) -> QueryAnalysis:
        self.calls.append(query)
        if not self.analyses:
            raise AssertionError("no fake query analysis remains")
        return self.analyses.pop(0)


class UnavailableQueryAnalyzer:
    """Analysis always fails, to prove the rule-based classifier still decides."""

    def __init__(self) -> None:
        self.calls = 0

    async def analyze(self, query: str) -> QueryAnalysis:
        self.calls += 1
        raise StructuredModelUnavailableError("query analyzer unavailable")


class FakeQueryPlanner:
    """Scripted plans, recording the resolved query and task it planned from.

    The recorded query is what the decomposition tests assert on: a plan must be
    built from the *standalone* question, not the raw follow-up, or a decomposed
    search would inherit the ambiguity contextual resolution just removed.
    """

    def __init__(self, *plans: QueryPlan) -> None:
        self.plans = list(plans)
        self.calls: list[tuple[str, QueryTask]] = []

    async def plan(self, query: str, analysis: QueryAnalysis) -> QueryPlan:
        self.calls.append((query, analysis.task))
        if not self.plans:
            raise AssertionError("no fake query plan remains")
        return self.plans.pop(0)


class UnavailableQueryPlanner:
    def __init__(self) -> None:
        self.calls = 0

    async def plan(self, query: str, analysis: QueryAnalysis) -> QueryPlan:
        self.calls += 1
        raise StructuredModelUnavailableError("query planner unavailable")


class RecordingOutputGuardrail:
    """Scripted output verdicts that record exactly what they were shown.

    ``drafts`` is the load-bearing part, the same way ``FakeChatModel.prompts``
    is: it proves validation saw the raw generated text, markers and all, rather
    than a version with the unresolvable citations already tidied away.
    """

    def __init__(self, *verdicts: OutputVerdict) -> None:
        self.verdicts = list(verdicts)
        self.drafts: list[str] = []
        self.source_counts: list[int] = []
        self.system_prompts: list[str] = []

    @property
    def calls(self) -> int:
        return len(self.drafts)

    async def validate(
        self,
        draft: str,
        sources: Sequence[SearchHit],
        *,
        system_prompt: str,
    ) -> OutputVerdict:
        self.drafts.append(draft)
        self.source_counts.append(len(sources))
        self.system_prompts.append(system_prompt)
        if not self.verdicts:
            raise AssertionError("no fake output verdict remains")
        return self.verdicts.pop(0)
