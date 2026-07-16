import uuid
from urllib.parse import quote

from fastapi import UploadFile
from fastapi.responses import StreamingResponse

from app.controller.document_controller.dto.document_dto import DocumentOut, DocumentUpdate
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
    workspace_id: uuid.UUID, folder_id: uuid.UUID | None, service: DocumentService
) -> list[DocumentOut]:
    documents = await service.list_documents(workspace_id, folder_id)
    return [DocumentOut.model_validate(document) for document in documents]


async def get_document(
    workspace_id: uuid.UUID, document_id: uuid.UUID, service: DocumentService
) -> DocumentOut:
    return DocumentOut.model_validate(await service.get(workspace_id, document_id))


async def download_document(
    workspace_id: uuid.UUID, document_id: uuid.UUID, service: DocumentService
) -> StreamingResponse:
    stream, file_name, mime_type = await service.open_stream(workspace_id, document_id)
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


async def list_trash(workspace_id: uuid.UUID, service: DocumentService) -> list[DocumentOut]:
    return [DocumentOut.model_validate(doc) for doc in await service.list_trash(workspace_id)]


async def delete_document_permanently(
    workspace_id: uuid.UUID, document_id: uuid.UUID, user: User, service: DocumentService
) -> None:
    await service.delete_permanently(user, workspace_id, document_id)
