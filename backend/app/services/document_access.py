"""Who a user is, as far as document visibility is concerned.

``DocumentRepository._accessible_condition`` is the SQL predicate; this is the
one function that decides what to feed it. Both halves matter and both are easy
to get subtly wrong:

- a workspace **owner** bypasses the predicate entirely (``None``), so an owner
  sees every document in their workspace, restricted or not;
- everyone else is (their user id, the teams they belong to *in this
  workspace*) - team membership is workspace-scoped, so a user's teams
  elsewhere must never widen what they can see here.

This lives on its own because two services now need it: ``DocumentService`` for
listings and ``SearchService`` for retrieval. Written twice it would eventually
drift, and the failure mode of drift here is a document surfacing in search that
its own listing hides.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.repository.team_repository import TeamRepository
from app.services.permission_service import PermissionService
from app.utils.rbac_catalog import OWNER

# (user_id, team_ids) for a filtered query; None means "no filter - see all"
AccessFilter = tuple[uuid.UUID, list[uuid.UUID]] | None


async def document_access_filter(
    session: AsyncSession, actor_id: uuid.UUID, workspace_id: uuid.UUID
) -> AccessFilter:
    """Access parameters for a visibility-filtered query.

    ``None`` = workspace owner, who needs no filter. Note this says nothing
    about whether the user is a member at all: the router's
    ``require_permission`` guard has already established that.
    """
    role = await PermissionService(session).workspace_role_name(actor_id, workspace_id)
    if role == OWNER:
        return None
    team_ids = await TeamRepository(session).team_ids_for_user(workspace_id, actor_id)
    return actor_id, team_ids
