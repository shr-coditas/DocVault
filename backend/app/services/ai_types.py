"""Shared immutable value types for document intelligence."""

import uuid
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Literal

NodeType = Literal[
    "document",
    "section",
    "heading",
    "paragraph",
    "list",
    "list_item",
    "table",
    "table_row",
    "table_cell",
    "caption",
    "code_block",
    "quote",
    "note",
    "warning",
    "figure",
    "page_break",
    "suppressed_header",
    "suppressed_footer",
]
BlockType = Literal["heading", "paragraph", "list_item", "table_row", "csv_row"]
ChunkType = Literal[
    "paragraph_chunk",
    "list_chunk",
    "list_item_chunk",
    "table_chunk",
    "table_row_group_chunk",
    "table_row_chunk",
    "code_chunk",
    "quote_chunk",
    "note_chunk",
    "warning_chunk",
    "figure_chunk",
    "fallback_text_chunk",
]


class SearchMode(StrEnum):
    SEMANTIC = "semantic"
    LEXICAL = "lexical"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """Location in the source or normalized extraction output."""

    page_number: int | None = None
    heading: str | None = None
    paragraph_index: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    row_start: int | None = None
    row_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    normalized_char_start: int | None = None
    normalized_char_end: int | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class ExtractedBlock:
    """Compatibility input for tests and the degraded line extractor."""

    text: str
    block_type: BlockType
    location: SourceLocation


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Legacy flat extraction accepted by ``ChunkingService`` during migration."""

    blocks: tuple[ExtractedBlock, ...]

    @property
    def is_empty(self) -> bool:
        return not any(block.text.strip() for block in self.blocks)

    @property
    def total_chars(self) -> int:
        return sum(len(block.text) for block in self.blocks)


@dataclass(frozen=True, slots=True)
class DocumentNode:
    """One node in the canonical, format-neutral document tree."""

    id: uuid.UUID
    logical_path: str
    node_type: NodeType
    parent_id: uuid.UUID | None
    ordinal: int
    text: str | None = None
    heading_level: int | None = None
    source_spans: tuple[SourceLocation, ...] = ()
    confidence: float = 1.0
    attributes: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExtractedArtifact:
    """Versioned structured extraction and its reproducibility metadata."""

    nodes: tuple[DocumentNode, ...]
    parser_name: str
    parser_version: str
    source_checksum: str
    metadata: dict[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    quality_metrics: dict[str, object] = field(default_factory=dict)
    raw_payload: dict[str, object] | None = None

    @property
    def root(self) -> DocumentNode:
        return next(node for node in self.nodes if node.parent_id is None)

    @property
    def is_empty(self) -> bool:
        return not any(
            node.text and node.text.strip()
            for node in self.nodes
            if node.node_type not in {"document", "section", "heading", "page_break"}
        )

    @property
    def total_chars(self) -> int:
        return sum(len(node.text or "") for node in self.nodes)

    @property
    def blocks(self) -> tuple[ExtractedBlock, ...]:
        """Compatibility projection used by the original extraction tests."""
        mapping: dict[NodeType, BlockType] = {
            "heading": "heading",
            "paragraph": "paragraph",
            "list_item": "list_item",
            "table_row": "table_row",
        }
        nodes_by_id = {node.id: node for node in self.nodes}
        rows = []
        for node in self.nodes:
            block_type = mapping.get(node.node_type)
            if block_type is None or not node.text:
                continue
            location = node.source_spans[0] if node.source_spans else SourceLocation()
            heading = location.heading
            parent_id = node.parent_id
            while heading is None and parent_id is not None:
                parent = nodes_by_id[parent_id]
                if parent.node_type in {"section", "heading"} and parent.text:
                    heading = parent.text
                    break
                parent_id = parent.parent_id
            if heading != location.heading:
                location = replace(location, heading=heading)
            rows.append(ExtractedBlock(node.text, block_type, location))
        return tuple(rows)


@dataclass(frozen=True, slots=True)
class ChunkSourceSpan:
    """Maps a substring of assembled chunk content back to its source."""

    sequence: int
    chunk_start: int
    chunk_end: int
    location: SourceLocation


def source_spans_from_json(rows: list[dict[str, object]]) -> tuple[ChunkSourceSpan, ...]:
    """Restore typed provenance from the JSONB representation."""
    spans: list[ChunkSourceSpan] = []
    for row in rows:
        raw_location = row.get("location")
        if isinstance(raw_location, dict):
            location_data = dict(raw_location)
            raw_bbox = location_data.get("bbox")
            if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
                location_data["bbox"] = tuple(float(value) for value in raw_bbox)
            location = SourceLocation(**location_data)
        else:
            location = SourceLocation()
        sequence = row.get("sequence")
        chunk_start = row.get("chunk_start")
        chunk_end = row.get("chunk_end")
        if not (
            isinstance(sequence, int)
            and isinstance(chunk_start, int)
            and isinstance(chunk_end, int)
        ):
            raise ValueError("invalid chunk source span")
        spans.append(ChunkSourceSpan(sequence, chunk_start, chunk_end, location))
    return tuple(spans)


@dataclass(frozen=True, slots=True)
class TextChunk:
    chunk_index: int
    content: str
    token_count: int
    heading: str | None
    source_spans: tuple[ChunkSourceSpan, ...]
    embedding_text: str = ""
    lexical_text: str = ""
    chunk_type: ChunkType = "paragraph_chunk"
    logical_key: str = ""
    structural_node_id: uuid.UUID | None = None
    parent_node_id: uuid.UUID | None = None
    ordinal_in_parent: int = 0
    heading_path: tuple[str, ...] = ()
    breadcrumb: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    content_hash: str = ""
    language: str = "und"
    metadata: dict[str, object] = field(default_factory=dict)


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
    token_count: int = 0
    heading: str | None = None
    source_spans: tuple[ChunkSourceSpan, ...] = ()
    mode: SearchMode = SearchMode.SEMANTIC
    chunk_type: ChunkType = "paragraph_chunk"
    breadcrumb: str | None = None
    logical_key: str = ""
    index_generation: int = 0
    scores: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    structural_node_id: uuid.UUID | None = None
    parent_node_id: uuid.UUID | None = None
    ordinal_in_parent: int = 0
    content_hash: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def page_numbers(self) -> tuple[int, ...]:
        return tuple(
            dict.fromkeys(
                span.location.page_number
                for span in self.source_spans
                if span.location.page_number is not None
            )
        )

    @property
    def page_number(self) -> int | None:
        return self.page_numbers[0] if self.page_numbers else None


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


class SafetyVerdict(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"
    UNCERTAIN = "uncertain"


class SafetyCategory(StrEnum):
    INSTRUCTION_OVERRIDE = "instruction_override"
    PROMPT_EXFILTRATION = "prompt_exfiltration"
    RAW_CONTEXT_EXFILTRATION = "raw_context_exfiltration"
    CROSS_TENANT_EXFILTRATION = "cross_tenant_exfiltration"
    JAILBREAK = "jailbreak"


class QueryTask(StrEnum):
    LOOKUP = "lookup"
    SUMMARY = "summary"
    COMPARISON = "comparison"
    MULTI_PART = "multi_part"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    safety: SafetyVerdict
    safety_categories: tuple[SafetyCategory, ...]
    intent: QueryIntent
    task: QueryTask
    confidence: float
    reason_code: str

    def __post_init__(self) -> None: #__post_init__ is a special method in Python that is called automatically after the __init__ method of a dataclass. It allows for additional initialization or validation of the dataclass fields after they have been set. In this case, it is used to validate the confidence and reason_code fields of the QueryAnalysis dataclass.
        if not 0 <= self.confidence <= 1:
            raise ValueError("query-analysis confidence must be between zero and one")
        if not self.reason_code:
            raise ValueError("query-analysis reason code must not be empty")


@dataclass(frozen=True, slots=True)
class QueryPlan:
    task: QueryTask
    search_queries: tuple[str, ...]
    required_aspects: tuple[str, ...] = ()
    needs_clarification: bool = False
    clarification_question: str | None = None

    def __post_init__(self) -> None:
        if len(self.search_queries) > 3:
            raise ValueError("query plans may contain at most three searches")
        if len(self.required_aspects) > 8:
            raise ValueError("query plans may contain at most eight required aspects")
        if any(not query.strip() or len(query) > 4000 for query in self.search_queries):
            raise ValueError("planned searches must be non-empty and at most 4000 characters")
        if self.needs_clarification:
            if self.search_queries or not self.clarification_question:
                raise ValueError("clarification plans require a question and no searches")
        elif not self.search_queries or self.clarification_question is not None:
            raise ValueError("retrieval plans require searches and no clarification question")


@dataclass(frozen=True, slots=True)
class EvidenceGrade:
    sufficient: bool
    relevant_source_numbers: tuple[int, ...] = ()
    missing_aspects: tuple[str, ...] = ()
    suggested_query: str | None = None
    conflict_detected: bool = False

    def __post_init__(self) -> None:
        if len(self.relevant_source_numbers) > 50 or any(
            number < 1 for number in self.relevant_source_numbers
        ):
            raise ValueError("evidence source numbers must be positive and bounded")
        if len(self.missing_aspects) > 8:
            raise ValueError("evidence grades may contain at most eight missing aspects")
        if self.suggested_query is not None and (
            not self.suggested_query.strip() or len(self.suggested_query) > 4000
        ):
            raise ValueError("suggested queries must be non-empty and at most 4000 characters")


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
class QueryExecutionContext:
    """Conversation-owned state loaded only after the raw query is safe."""

    document_ids: tuple[uuid.UUID, ...] | None
    history: tuple[ConversationTurn, ...] = ()
    unavailable_documents: tuple[UnavailableDocument, ...] = ()

    @property
    def scope_degraded(self) -> bool:
        return bool(self.unavailable_documents)


@dataclass(frozen=True, slots=True)
class Citation:
    marker: int
    document_id: uuid.UUID
    document_title: str
    chunk_id: uuid.UUID
    heading: str | None = None
    source_spans: tuple[ChunkSourceSpan, ...] = ()
    index_generation: int = 0
    logical_key: str = ""

    @property
    def page_numbers(self) -> tuple[int, ...]:
        return tuple(
            dict.fromkeys(
                span.location.page_number
                for span in self.source_spans
                if span.location.page_number is not None
            )
        )

    @property
    def page_number(self) -> int | None:
        return self.page_numbers[0] if self.page_numbers else None


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
    # None means evidence was never graded - the legacy path, and the graph with
    # no grader configured. That is a different fact from "graded and found
    # wanting", which is what lets the conversation layer label an unsupported
    # answer honestly instead of blaming the generation provider.
    evidence: EvidenceGrade | None = None
    # 6D. What the question was taken to be, and how it was searched. Null on
    # the legacy path and whenever no planner ran, for the same reason `evidence`
    # is: "not planned" and "planned as a single lookup" are different facts.
    task: QueryTask | None = None
    plan: QueryPlan | None = None
    # 6E. Null means the draft was never validated; a verdict that did not pass
    # means a draft existed and was rejected, which is why `answer` is null here
    # without the provider having failed. Rejected draft text is never carried.
    output_verdict: OutputVerdict | None = None
    generation_attempts: int = 0

    @property
    def sources_conflict(self) -> bool:
        """Whether grading found the authorized sources contradicting each other."""
        return self.evidence is not None and self.evidence.conflict_detected


IndexStatus = Literal["indexed", "skipped", "failed"]


@dataclass(frozen=True, slots=True)
class IndexOutcome:
    document_id: uuid.UUID
    status: IndexStatus
    chunk_count: int = 0
    detail: str | None = None
