import uuid
from typing import Annotated

from fastapi import APIRouter, Depends

from app.controller.query_controller import query_controller
from app.controller.query_controller.dto.query_dto import QueryIn, QueryOut
from app.dependencies import (
    ChatModelDep,
    DbSession,
    EmbedderDep,
    RerankerDep,
    SupervisorDep,
    require_permission,
)
from app.models.user import User
from app.services.answer_service import AnswerService
from app.services.query_service import QueryService
from app.services.search_service import SearchService
from app.utils.rbac_catalog import Perm

router = APIRouter(prefix="/workspaces/{workspace_id}/query", tags=["query"])


def get_query_service(
    db: DbSession,
    embedder: EmbedderDep,
    reranker: RerankerDep,
    model: ChatModelDep,
    supervisor: SupervisorDep,
) -> QueryService:
    # SearchService is injected rather than reached for, so the intent gate can
    # be tested with a search double that records whether it was called at all.
    # AnswerService likewise: the ChatModel seam is what keeps the suite offline,
    # and it is the only way to assert *which* passages reached the model.
    return QueryService(
        SearchService(db, embedder, reranker=reranker),
        answers=AnswerService(model),
        supervisor=supervisor,
    )


ServiceDep = Annotated[QueryService, Depends(get_query_service)]

# Same capability as /search: this endpoint retrieves document text, so it is a
# document read. Per-document visibility is enforced inside the retrieval, and
# the intent gate never widens what may be seen - it only decides whether to look.
CanRead = Annotated[User, Depends(require_permission(Perm.DOCUMENT_READ))]


# POST, not GET: the question goes in the body rather than the URL, which keeps it
# out of proxy logs and browser history - the open issue /search still carries.
@router.post("")
async def submit_query(
    workspace_id: uuid.UUID,
    data: QueryIn,
    user: CanRead,
    service: ServiceDep,
) -> QueryOut:
    """Guardrail, classify, retrieve if warranted, and answer over what returned.

    Always 200 with a decision - a blocked or declined query is a normal outcome
    of the pipeline, not a client error. Refusals stay fixed sentences rather
    than generated prose: there is nothing retrieved to ground them in, and a
    model asked to phrase a refusal is a model invited to negotiate it.
    """
    return await query_controller.handle_query(workspace_id, data, user, service)
