import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from fastapi.responses import StreamingResponse

from app.controller.document_controller import document_controller
from app.controller.document_controller.dto.document_dto import (
    DocumentOut,
    DocumentUpdate,
    GrantCreate,
    GrantOut,
    VisibilityUpdate,
)
from app.dependencies import DbSession, StorageDep, require_permission
from app.models.document_grant import PrincipalType
from app.models.user import User
from app.services.document_service import DocumentService
from app.utils.rbac_catalog import Perm

router = APIRouter(prefix="/workspaces/{workspace_id}/documents", tags=["documents"])


def get_document_service(db: DbSession, storage: StorageDep) -> DocumentService:
    return DocumentService(db, storage)


ServiceDep = Annotated[DocumentService, Depends(get_document_service)]

CanRead = Annotated[User, Depends(require_permission(Perm.DOCUMENT_READ))]
CanCreate = Annotated[User, Depends(require_permission(Perm.DOCUMENT_CREATE))]
CanUpdate = Annotated[User, Depends(require_permission(Perm.DOCUMENT_UPDATE))]
CanDelete = Annotated[User, Depends(require_permission(Perm.DOCUMENT_DELETE))]
CanShare = Annotated[User, Depends(require_permission(Perm.DOCUMENT_SHARE))]


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    workspace_id: uuid.UUID,
    user: CanCreate,
    service: ServiceDep,
    file: Annotated[UploadFile, File()],
    folder_id: Annotated[uuid.UUID | None, Form()] = None,
    title: Annotated[str | None, Form()] = None,
) -> DocumentOut:
    return await document_controller.upload_document(
        workspace_id, folder_id, title, file, user, service
    )


@router.get("")
async def list_documents(
    workspace_id: uuid.UUID,
    user: CanRead,
    service: ServiceDep,
    folder_id: Annotated[uuid.UUID | None, Query()] = None,
    scope: Annotated[Literal["folder", "all"], Query()] = "folder",
) -> list[DocumentOut]:
    """``scope=folder`` (default) lists one folder, where a missing ``folder_id``
    means the workspace root; ``scope=all`` spans every folder and ignores it."""
    return await document_controller.list_documents(workspace_id, folder_id, scope, user, service)


# NOTE: /trash must be declared before /{document_id} so it isn't parsed as a uuid
@router.get("/trash")
async def list_trash(
    workspace_id: uuid.UUID, user: CanDelete, service: ServiceDep
) -> list[DocumentOut]:
    return await document_controller.list_trash(workspace_id, user, service)


@router.get("/{document_id}")
async def get_document(
    workspace_id: uuid.UUID, document_id: uuid.UUID, user: CanRead, service: ServiceDep
) -> DocumentOut:
    return await document_controller.get_document(workspace_id, document_id, user, service)


@router.get("/{document_id}/download")
async def download_document(
    workspace_id: uuid.UUID, document_id: uuid.UUID, user: CanRead, service: ServiceDep
) -> StreamingResponse:
    return await document_controller.download_document(workspace_id, document_id, user, service)


@router.patch("/{document_id}")
async def update_document(
    workspace_id: uuid.UUID,
    document_id: uuid.UUID,
    data: DocumentUpdate,
    user: CanUpdate,
    service: ServiceDep,
) -> DocumentOut:
    return await document_controller.update_document(workspace_id, document_id, data, user, service)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def trash_document(
    workspace_id: uuid.UUID, document_id: uuid.UUID, user: CanDelete, service: ServiceDep
) -> None:
    await document_controller.trash_document(workspace_id, document_id, user, service)


@router.post("/{document_id}/restore")
async def restore_document(
    workspace_id: uuid.UUID, document_id: uuid.UUID, user: CanDelete, service: ServiceDep
) -> DocumentOut:
    return await document_controller.restore_document(workspace_id, document_id, user, service)


@router.delete("/{document_id}/permanent", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document_permanently(
    workspace_id: uuid.UUID, document_id: uuid.UUID, user: CanDelete, service: ServiceDep
) -> None:
    await document_controller.delete_document_permanently(workspace_id, document_id, user, service)


@router.put("/{document_id}/visibility")
async def set_visibility(
    workspace_id: uuid.UUID,
    document_id: uuid.UUID,
    data: VisibilityUpdate,
    user: CanShare,
    service: ServiceDep,
) -> DocumentOut:
    return await document_controller.set_visibility(workspace_id, document_id, data, user, service)


@router.get("/{document_id}/grants")
async def list_grants(
    workspace_id: uuid.UUID, document_id: uuid.UUID, user: CanShare, service: ServiceDep
) -> list[GrantOut]:
    return await document_controller.list_grants(workspace_id, document_id, user, service)


@router.post("/{document_id}/grants", status_code=status.HTTP_201_CREATED)
async def add_grant(
    workspace_id: uuid.UUID,
    document_id: uuid.UUID,
    data: GrantCreate,
    user: CanShare,
    service: ServiceDep,
) -> GrantOut:
    return await document_controller.add_grant(workspace_id, document_id, data, user, service)


@router.delete(
    "/{document_id}/grants/{principal_type}/{principal_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_grant(
    workspace_id: uuid.UUID,
    document_id: uuid.UUID,
    principal_type: PrincipalType,
    principal_id: uuid.UUID,
    user: CanShare,
    service: ServiceDep,
) -> None:
    await document_controller.remove_grant(
        workspace_id, document_id, principal_type, principal_id, user, service
    )
