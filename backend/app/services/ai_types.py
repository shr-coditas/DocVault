"""Shared immutable value types for document intelligence."""

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

ChunkType = Literal["paragraph", "list", "table", "code"]


class SearchMode(StrEnum):
    SEMANTIC = "semantic"
    LEXICAL = "lexical"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class MarkdownBlock:
    block_type: ChunkType
    content: str
    section_path: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractedMarkdown:
    blocks: tuple[MarkdownBlock, ...]

    @property
    def is_empty(self) -> bool:
        return not any(block.content.strip() for block in self.blocks)

    @property
    def total_chars(self) -> int:
        return sum(len(block.content) for block in self.blocks)


@dataclass(frozen=True, slots=True)
class TextChunk:
    chunk_index: int
    chunk_type: ChunkType
    section_path: str | None
    content: str


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    semantic: float | None = None
    lexical: float | None = None
    fusion: float | None = None
    rerank: float | None = None


@dataclass(frozen=True, slots=True)
class SearchHit:
    document_id: uuid.UUID
    document_title: str
    file_name: str
    chunk_id: uuid.UUID
    chunk_index: int
    content: str
    score: float
    mode: SearchMode = SearchMode.SEMANTIC
    chunk_type: ChunkType = "paragraph"
    section_path: str | None = None
    scores: ScoreBreakdown = field(default_factory=ScoreBreakdown)


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: str
    limit: int
    semantic_min_score: float | None
    hits: tuple[SearchHit, ...]
    mode: SearchMode = SearchMode.SEMANTIC


class QueryIntent(StrEnum):
    DOCUMENT_QUESTION = "document_question"
    CHITCHAT = "chitchat"
    OUT_OF_SCOPE = "out_of_scope"
    PROMPT_INJECTION = "prompt_injection"


class QueryDecision(StrEnum):
    RETRIEVE = "retrieve"
    ANSWER_DIRECTLY = "answer_directly"
    DECLINE = "decline"
    BLOCK = "block"
    CLARIFY = "clarify"
    SCOPE_UNAVAILABLE = "scope_unavailable"


class OutputIssue(StrEnum):
    SYSTEM_PROMPT_DISCLOSURE = "system_prompt_disclosure"
    CONFIGURATION_DISCLOSURE = "configuration_disclosure"
    HIDDEN_CHARACTERS = "hidden_characters"
    UNRESOLVABLE_CITATIONS = "unresolvable_citations"
    MISSING_CITATIONS = "missing_citations"
    EXCESSIVE_SOURCE_COPY = "excessive_source_copy"


_SECURITY_OUTPUT_ISSUES = frozenset(
    {
        OutputIssue.SYSTEM_PROMPT_DISCLOSURE,
        OutputIssue.CONFIGURATION_DISCLOSURE,
        OutputIssue.HIDDEN_CHARACTERS,
    }
)


@dataclass(frozen=True, slots=True)
class OutputVerdict:
    issues: tuple[OutputIssue, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.issues

    @property
    def security_failure(self) -> bool:
        return any(issue in _SECURITY_OUTPUT_ISSUES for issue in self.issues)

    @property
    def can_regenerate(self) -> bool:
        return bool(self.issues) and not self.security_failure


@dataclass(frozen=True, slots=True)
class GuardrailVerdict:
    name: str
    passed: bool
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class GuardrailOutcome:
    verdicts: tuple[GuardrailVerdict, ...]

    @property
    def passed(self) -> bool:
        return all(verdict.passed for verdict in self.verdicts)

    @property
    def failure(self) -> GuardrailVerdict | None:
        return next((verdict for verdict in self.verdicts if not verdict.passed), None)


@dataclass(frozen=True, slots=True)
class IntentJudgement:
    intent: QueryIntent
    confidence: float
    reason: str


class ContextReason(StrEnum):
    REWRITTEN = "rewritten"
    UNCHANGED = "unchanged"
    AMBIGUOUS = "ambiguous"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    user_message: str
    assistant_message: str


@dataclass(frozen=True, slots=True)
class ContextResolution:
    standalone_query: str
    used_history: bool
    needs_clarification: bool
    reason_code: ContextReason


@dataclass(frozen=True, slots=True)
class UnavailableDocument:
    document_id: uuid.UUID
    title: str
    file_name: str


@dataclass(frozen=True, slots=True)
class DocumentBrief:
    """Access-safe routing context for one selected document."""

    document_id: uuid.UUID
    title: str
    summary: str


@dataclass(frozen=True, slots=True)
class QueryExecutionContext:
    """Conversation-owned state loaded only after the raw query is safe."""

    document_ids: tuple[uuid.UUID, ...] | None
    history: tuple[ConversationTurn, ...] = ()
    unavailable_documents: tuple[UnavailableDocument, ...] = ()
    document_summaries: tuple[DocumentBrief, ...] = ()
    summaries_complete: bool = False

    @property
    def scope_degraded(self) -> bool:
        return bool(self.unavailable_documents)


@dataclass(frozen=True, slots=True)
class Citation:
    marker: int
    document_id: uuid.UUID
    document_title: str
    chunk_id: uuid.UUID
    section_path: str | None = None
    chunk_type: ChunkType = "paragraph"


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    text: str
    model: str
    citations: tuple[Citation, ...] = ()
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class AnswerDraft:
    """Generated text as the model wrote it, before citation resolution.

    6E validates this rather than the finished ``GeneratedAnswer``. Resolution
    *drops* markers that point at no supplied source, so a guardrail running
    after it would inspect a tidy bibliography and never learn that the model
    invented ``[7]``. The invented marker is exactly the signal worth catching,
    which is why drafting and resolution had to become two steps.
    """

    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class GenerationFailure(StrEnum):
    NO_SOURCES = "no_sources"
    PROVIDER_UNAVAILABLE = "provider_unavailable"


@dataclass(frozen=True, slots=True)
class AnswerAttempt:
    """Internal generation result, including exactly what reached the model.

    ``selected_sources`` is deliberately kept out of the public query DTO. The
    durable conversation layer needs it to distinguish retrieved hits from the
    passages that could have influenced generated prose, without adding another
    chunk-content field to every stateless ``/query`` response.
    """

    answer: GeneratedAnswer | None
    selected_sources: tuple[SearchHit, ...] = ()
    failure: GenerationFailure | None = None


@dataclass(frozen=True, slots=True)
class DraftAttempt:
    """``AnswerAttempt`` before citations are resolved; same failure semantics."""

    draft: AnswerDraft | None
    selected_sources: tuple[SearchHit, ...] = ()
    failure: GenerationFailure | None = None


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    query: str
    intent: QueryIntent
    confidence: float
    decision: QueryDecision
    reason: str
    guardrails: GuardrailOutcome
    retrieval_performed: bool
    hits: tuple[SearchHit, ...] = ()
    message: str | None = None
    answer: GeneratedAnswer | None = None
    selected_sources: tuple[SearchHit, ...] = ()
    context_resolution: ContextResolution | None = None
    scope_degraded: bool = False
    unavailable_documents: tuple[UnavailableDocument, ...] = ()
    # None means nobody judged whether the passages answered the question - the
    # linear pipeline, and any turn that never retrieved. That is a different
    # fact from "judged and found wanting", and it is what lets the conversation
    # layer record an unsupported answer honestly instead of blaming the
    # generation provider for a silence it had nothing to do with.
    evidence_sufficient: bool | None = None
    # None means the draft was never checked. A verdict that did not pass means
    # a draft existed and was thrown away, which is why `answer` is None here
    # without the provider having failed. The rejected text is never carried.
    output_verdict: OutputVerdict | None = None
    generation_attempts: int = 0


IndexStatus = Literal["indexed", "skipped", "failed"]


@dataclass(frozen=True, slots=True)
class IndexOutcome:
    document_id: uuid.UUID
    status: IndexStatus
    chunk_count: int = 0
    detail: str | None = None
