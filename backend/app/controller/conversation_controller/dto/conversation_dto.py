import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.controller.search_controller.dto.search_dto import ChunkSourceSpanOut
from app.models.conversation import ConversationScope, MessageKind, MessageRole, MessageStatus


class ConversationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope_mode: ConversationScope
    document_ids: list[uuid.UUID] | None = Field(default=None, max_length=10)

    @model_validator(mode="after")
    def validate_scope(self) -> "ConversationCreate":
        document_ids = self.document_ids or []
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document_ids must be unique")
        if self.scope_mode is ConversationScope.WORKSPACE and document_ids:
            raise ValueError("workspace scope must not include document_ids")
        if self.scope_mode is ConversationScope.SELECTED and not document_ids:
            raise ValueError("selected scope requires at least one document_id")
        return self


class ConversationDocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: uuid.UUID
    position: int
    title: str
    file_name: str


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    workspace_id: uuid.UUID
    title: str
    scope_mode: ConversationScope
    documents: list[ConversationDocumentOut]
    created_at: datetime
    updated_at: datetime


class ConversationPageOut(BaseModel):
    items: list[ConversationOut]
    next_cursor: str | None


class ConversationMessageCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=4000)
    client_message_id: uuid.UUID


class ConversationMessageSourceOut(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    chunk_id: uuid.UUID
    resolved_chunk_id: uuid.UUID | None
    relocated: bool
    index_generation: int
    logical_key: str
    heading: str | None
    breadcrumb: str | None
    page_numbers: list[int]
    source_spans: list[ChunkSourceSpanOut]
    retrieval_rank: int
    supplied_to_model: bool
    citation_marker: int | None


class UnavailableConversationDocumentOut(BaseModel):
    document_id: uuid.UUID
    title: str
    file_name: str


class ConversationMessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    turn_id: uuid.UUID
    sequence: int
    role: MessageRole
    status: MessageStatus
    kind: MessageKind | Literal["redacted"] | None
    content: str | None
    context_eligible: bool
    redacted: bool = False
    sources: list[ConversationMessageSourceOut] = Field(default_factory=list)
    scope_degraded: bool = False
    unavailable_documents: list[UnavailableConversationDocumentOut] = Field(default_factory=list)
    client_message_id: uuid.UUID | None
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    created_at: datetime
    updated_at: datetime


class ConversationMessagePageOut(BaseModel):
    items: list[ConversationMessageOut]
    next_after_sequence: int | None


class ConversationTurnOut(BaseModel):
    turn_id: uuid.UUID
    status: MessageStatus
    messages: list[ConversationMessageOut]
    status_url: str
    retry_after_seconds: int | None = None
    scope_degraded: bool = False
    unavailable_documents: list[UnavailableConversationDocumentOut] = Field(default_factory=list)
