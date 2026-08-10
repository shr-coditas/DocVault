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


IndexStatus = Literal["indexed", "skipped", "failed"]


@dataclass(frozen=True, slots=True)
class IndexOutcome:
    document_id: uuid.UUID
    status: IndexStatus
    chunk_count: int = 0
    detail: str | None = None
