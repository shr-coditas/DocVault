import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from fastapi.responses import StreamingResponse

from app.controller.document_controller import document_controller
from app.controller.document_controller.dto.document_dto import DocumentOut
from app.dependencies import DbSession, require_permission
from app.models.user import User
from app.services.document_service import DocumentService
from app.services.storage_service import StorageService
from app.utils.rbac_catalog import Perm

router = APIRouter(prefix="/workspaces/{workspace_id}/documents", tags=["documents"])


def get_storage_service() -> StorageService:
    return StorageService()


StorageDep = Annotated[StorageService, Depends(get_storage_service)]


def get_document_service(db: DbSession, storage: StorageDep) -> DocumentService:
    return DocumentService(db, storage)


ServiceDep = Annotated[DocumentService, Depends(get_document_service)]

CanRead = Annotated[User, Depends(require_permission(Perm.DOCUMENT_READ))]
CanCreate = Annotated[User, Depends(require_permission(Perm.DOCUMENT_CREATE))]


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
) -> list[DocumentOut]:
    return await document_controller.list_documents(workspace_id, folder_id, service)


@router.get("/{document_id}")
async def get_document(
    workspace_id: uuid.UUID, document_id: uuid.UUID, user: CanRead, service: ServiceDep
) -> DocumentOut:
    return await document_controller.get_document(workspace_id, document_id, service)


@router.get("/{document_id}/download")
async def download_document(
    workspace_id: uuid.UUID, document_id: uuid.UUID, user: CanRead, service: ServiceDep
) -> StreamingResponse:
    return await document_controller.download_document(workspace_id, document_id, service)
