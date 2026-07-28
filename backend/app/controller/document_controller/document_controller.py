import uuid
from urllib.parse import quote

from fastapi import UploadFile
from fastapi.responses import StreamingResponse

from app.controller.document_controller.dto.document_dto import (
    DocumentOut,
    DocumentUpdate,
    GrantCreate,
    GrantOut,
    VisibilityUpdate,
)
from app.models.document_grant import PrincipalType
from app.models.user import User
from app.services.document_service import DocumentService


async def upload_document(
    workspace_id: uuid.UUID,
    folder_id: uuid.UUID | None,
    title: str | None,
    upload: UploadFile,
    user: User,
    service: DocumentService,
) -> DocumentOut:
    document = await service.upload(user, workspace_id, folder_id, upload, title)
    return DocumentOut.model_validate(document)


async def list_documents(
    workspace_id: uuid.UUID,
    folder_id: uuid.UUID | None,
    scope: str,
    user: User,
    service: DocumentService,
) -> list[DocumentOut]:
    documents = await service.list_documents(
        user, workspace_id, folder_id, every_folder=scope == "all"
    )
    return [DocumentOut.model_validate(document) for document in documents]


async def get_document(
    workspace_id: uuid.UUID, document_id: uuid.UUID, user: User, service: DocumentService
) -> DocumentOut:
    return DocumentOut.model_validate(await service.get(user, workspace_id, document_id))


async def download_document(
    workspace_id: uuid.UUID, document_id: uuid.UUID, user: User, service: DocumentService
) -> StreamingResponse:
    stream, file_name, mime_type = await service.open_stream(user, workspace_id, document_id)
    # RFC 5987 filename* so non-ASCII names survive the header
    disposition = f"attachment; filename*=UTF-8''{quote(file_name)}"
    return StreamingResponse(
        stream, media_type=mime_type, headers={"Content-Disposition": disposition}
    )


async def update_document(
    workspace_id: uuid.UUID,
    document_id: uuid.UUID,
    data: DocumentUpdate,
    user: User,
    service: DocumentService,
) -> DocumentOut:
    return DocumentOut.model_validate(await service.update(user, workspace_id, document_id, data))


async def trash_document(
    workspace_id: uuid.UUID, document_id: uuid.UUID, user: User, service: DocumentService
) -> None:
    await service.trash(user, workspace_id, document_id)


async def restore_document(
    workspace_id: uuid.UUID, document_id: uuid.UUID, user: User, service: DocumentService
) -> DocumentOut:
    return DocumentOut.model_validate(await service.restore(user, workspace_id, document_id))


async def list_trash(
    workspace_id: uuid.UUID, user: User, service: DocumentService
) -> list[DocumentOut]:
    documents = await service.list_trash(user, workspace_id)
    return [DocumentOut.model_validate(doc) for doc in documents]


async def delete_document_permanently(
    workspace_id: uuid.UUID, document_id: uuid.UUID, user: User, service: DocumentService
) -> None:
    await service.delete_permanently(user, workspace_id, document_id)


async def set_visibility(
    workspace_id: uuid.UUID,
    document_id: uuid.UUID,
    data: VisibilityUpdate,
    user: User,
    service: DocumentService,
) -> DocumentOut:
    document = await service.set_visibility(user, workspace_id, document_id, data.visibility)
    return DocumentOut.model_validate(document)


async def list_grants(
    workspace_id: uuid.UUID, document_id: uuid.UUID, user: User, service: DocumentService
) -> list[GrantOut]:
    grants = await service.list_grants(user, workspace_id, document_id)
    return [GrantOut.model_validate(grant) for grant in grants]


async def add_grant(
    workspace_id: uuid.UUID,
    document_id: uuid.UUID,
    data: GrantCreate,
    user: User,
    service: DocumentService,
) -> GrantOut:
    grant = await service.add_grant(
        user, workspace_id, document_id, data.principal_type, data.principal_id
    )
    return GrantOut.model_validate(grant)


async def remove_grant(
    workspace_id: uuid.UUID,
    document_id: uuid.UUID,
    principal_type: PrincipalType,
    principal_id: uuid.UUID,
    user: User,
    service: DocumentService,
) -> None:
    await service.remove_grant(user, workspace_id, document_id, principal_type, principal_id)
