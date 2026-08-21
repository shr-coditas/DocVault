import uuid

from pydantic import BaseModel, ConfigDict

from app.services.ai_types import SearchMode


class SourceLocationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    page_number: int | None
    heading: str | None
    paragraph_index: int | None
    line_start: int | None
    line_end: int | None
    row_start: int | None
    row_end: int | None
    char_start: int | None
    char_end: int | None
    normalized_char_start: int | None = None
    normalized_char_end: int | None = None
    bbox: tuple[float, float, float, float] | None = None


class ChunkSourceSpanOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    sequence: int
    chunk_start: int
    chunk_end: int
    location: SourceLocationOut


class ScoreBreakdownOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    semantic: float | None
    lexical: float | None
    fusion: float | None
    rerank: float | None


class SearchHitOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: uuid.UUID
    document_title: str
    file_name: str
    chunk_id: uuid.UUID
    chunk_index: int
    content: str
    score: float
    page_number: int | None
    page_numbers: list[int]
    heading: str | None
    source_spans: list[ChunkSourceSpanOut]
    mode: SearchMode
    chunk_type: str
    breadcrumb: str | None
    logical_key: str
    scores: ScoreBreakdownOut


class SearchResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    query: str
    mode: SearchMode
    limit: int
    semantic_min_score: float | None
    count: int
    hits: list[SearchHitOut]
