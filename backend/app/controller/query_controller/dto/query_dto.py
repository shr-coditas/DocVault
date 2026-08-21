import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Reused rather than redeclared: a retrieved chunk is the same thing here as it
# is on /search, and two Pydantic models for one concept drift the moment one of
# them gains a field. The direction of the import is the tolerable half of the
# trade - this DTO depends on the search DTO, never the reverse.
from app.controller.search_controller.dto.search_dto import ChunkSourceSpanOut, SearchHitOut
from app.services.ai_types import QueryDecision, QueryIntent, SearchMode


class QueryIn(BaseModel):
    """A question, plus the same retrieval knobs /search exposes.

    They are accepted even for queries that will not retrieve: the caller cannot
    know in advance which intent theirs will be classified as, and rejecting
    ``limit`` on a greeting would be a puzzle rather than a safeguard.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    limit: int | None = Field(default=None, ge=1, le=50)
    semantic_min_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    retrieval_mode: SearchMode = SearchMode.HYBRID
    document_id: uuid.UUID | None = None
    document_ids: list[uuid.UUID] | None = Field(default=None, max_length=10)

    @model_validator(mode="after")
    def validate_scope_and_scores(self) -> "QueryIn":
        if self.document_id is not None and self.document_ids:
            raise ValueError("document_id and document_ids are mutually exclusive")
        if self.retrieval_mode is SearchMode.LEXICAL and self.semantic_min_score is not None:
            raise ValueError("semantic score parameters do not apply to lexical mode")
        return self


class GuardrailVerdictOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    passed: bool
    detail: str | None


class CitationOut(BaseModel):
    """One bracketed marker the model wrote, resolved to the chunk behind it.

    ``marker`` is the number as it appears in ``answer.text``, so a client can
    turn "[2]" into a link without re-parsing the prose. Markers that did not
    resolve to a supplied source were dropped before this point.
    """

    model_config = ConfigDict(from_attributes=True)

    marker: int
    document_id: uuid.UUID
    document_title: str
    chunk_id: uuid.UUID
    page_number: int | None
    page_numbers: list[int]
    heading: str | None
    source_spans: list[ChunkSourceSpanOut]
    logical_key: str


class GeneratedAnswerOut(BaseModel):
    """The generated answer, with what produced it and what it cost.

    ``model`` and the token counts are exposed rather than logged-only because
    the evaluation slice compares runs across providers, and a number you have
    to grep the logs for is a number nobody compares. Token counts are null when
    the provider does not report them - absent, not zero.
    """

    model_config = ConfigDict(from_attributes=True)

    text: str
    model: str
    citations: list[CitationOut]
    input_tokens: int | None
    output_tokens: int | None


class QueryOut(BaseModel):
    """The pipeline's decision, and the evidence for it.

    Every field is here so a developer can answer "why did that happen?" without
    reading the logs: which checks ran, what the query was taken to be, whether
    that justified a retrieval, what came back, and what was generated over it.

    ``answer`` is null whenever generation did not happen or did not succeed -
    no retrieval, nothing retrieved, or the provider unavailable - and in each
    of those cases ``message`` explains which. A caller renders ``answer.text``
    when present and falls back to ``message`` plus ``hits`` when it is not.
    """

    model_config = ConfigDict(from_attributes=True)

    intent: QueryIntent
    confidence: float
    decision: QueryDecision
    reason: str
    guardrails: list[GuardrailVerdictOut]
    # stated, not inferred from an empty `hits`: retrieving nothing and never
    # retrieving are different events, and only one of them cost anything
    retrieval_performed: bool
    hits: list[SearchHitOut]
    message: str | None
    answer: GeneratedAnswerOut | None = None
