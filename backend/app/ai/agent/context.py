"""Everything a node needs that must not be reachable from graph state."""

import uuid
from dataclasses import dataclass

from app.ai.agent.supervisor import Supervisor
from app.config import Settings
from app.models.user import User
from app.services.ai_types import SearchMode, UnavailableDocument
from app.services.answer_service import AnswerService
from app.services.search_service import SearchService


@dataclass(slots=True)
class AgentContext:
    """Request-scoped services, and the scope this turn is allowed to read.

    LangGraph hands this to every node as ``runtime.context``. It is deliberately
    not graph state: state is what the supervisor reads and writes, and the
    supervisor must not be able to change who is asking or which documents are
    within reach. A rewritten search can look elsewhere in what this actor may
    already read, and nowhere else.
    """

    actor: User
    workspace_id: uuid.UUID
    search: SearchService
    answers: AnswerService
    supervisor: Supervisor
    settings: Settings

    # The conversation's pinned documents, already narrowed to the ones this
    # actor can still open, or None for a workspace-wide question.
    document_ids: tuple[uuid.UUID, ...] | None = None
    # Pinned documents the actor has lost access to. Retrieval already excludes
    # them; these are kept so the turn can say so rather than answer from a
    # quietly smaller corpus.
    unavailable_documents: tuple[UnavailableDocument, ...] = ()

    document_id: uuid.UUID | None = None
    limit: int | None = None
    semantic_min_score: float | None = None
    mode: SearchMode = SearchMode.HYBRID

    @property
    def scope_degraded(self) -> bool:
        return bool(self.unavailable_documents)
