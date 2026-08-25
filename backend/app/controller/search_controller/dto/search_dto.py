import uuid

from pydantic import BaseModel, ConfigDict

from app.services.ai_types import SearchMode


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
    mode: SearchMode
    chunk_type: str
    section_path: str | None
    scores: ScoreBreakdownOut


class SearchResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    query: str
    mode: SearchMode
    limit: int
    semantic_min_score: float | None
    count: int
    hits: list[SearchHitOut]
