import hashlib
import mimetypes
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from uuid6 import uuid7

from app.config import get_settings
from app.controller.document_controller.dto.document_dto import DocumentUpdate
from app.exceptions import NotFoundError, PayloadTooLargeError
from app.models.document import Document
from app.models.user import User
from app.repository.document_repository import DocumentRepository
from app.repository.folder_repository import FolderRepository
from app.services.audit_service import AuditService
from app.services.storage_service import CHUNK_SIZE, StorageService, document_key

DEFAULT_MIME = "application/octet-stream"


class DocumentService:
    def __init__(self, session: AsyncSession, storage: StorageService) -> None:
        self.session = session
        self.storage = storage
        self.repository = DocumentRepository(session)
        self.folders = FolderRepository(session)
        self.audit = AuditService(session)
        self.settings = get_settings()

    async def upload(
        self,
        actor: User,
        workspace_id: uuid.UUID,
        folder_id: uuid.UUID | None,
        upload: UploadFile,
        title: str | None,
    ) -> Document:
        if folder_id is not None:
            await self._require_folder(workspace_id, folder_id)

        file_name = upload.filename or "file"
        data, size, checksum = await self._read_capped(upload)
        mime_type = mimetypes.guess_type(file_name)[0] or upload.content_type or DEFAULT_MIME

        document_id = uuid7()
        key = document_key(str(workspace_id), str(document_id), 1, file_name)
        document = Document(
            id=document_id,
            workspace_id=workspace_id,
            folder_id=folder_id,
            owner_id=actor.id,
            title=title or file_name,
            file_name=file_name,
            mime_type=mime_type,
            size_bytes=size,
            checksum_sha256=checksum,
            storage_key=key,
        )
        self.repository.add(document)
        # bytes first: if the object store write fails we never commit the row
        await self.storage.put_bytes(key, data, mime_type)
        self.audit.record(
            action="document.uploaded",
            resource_type="document",
            resource_id=document_id,
            workspace_id=workspace_id,
            actor_id=actor.id,
            file_name=file_name,
            size_bytes=size,
        )
        await self.session.commit()
        await self.session.refresh(document)
        return document

    async def get(self, workspace_id: uuid.UUID, document_id: uuid.UUID) -> Document:
        """Active documents only — trashed ones are invisible here (404)."""
        document = await self._get_any(workspace_id, document_id)
        if document.deleted_at is not None:
            raise NotFoundError("document not found")
        return document

    async def list_documents(
        self, workspace_id: uuid.UUID, folder_id: uuid.UUID | None
    ) -> list[Document]:
        return await self.repository.list_in_workspace(workspace_id, folder_id)

    async def open_stream(
        self, workspace_id: uuid.UUID, document_id: uuid.UUID
    ) -> tuple[AsyncIterator[bytes], str, str]:
        document = await self.get(workspace_id, document_id)
        return self.storage.stream(document.storage_key), document.file_name, document.mime_type

    async def update(
        self, actor: User, workspace_id: uuid.UUID, document_id: uuid.UUID, data: DocumentUpdate
    ) -> Document:
        """Rename (title) and/or move (folder_id; None = workspace root)."""
        document = await self.get(workspace_id, document_id)
        changes = data.model_dump(exclude_unset=True)
        if not changes:
            return document

        if "folder_id" in changes and changes["folder_id"] is not None:
            await self._require_folder(workspace_id, changes["folder_id"])
        if "folder_id" in changes:
            document.folder_id = changes["folder_id"]
        if changes.get("title"):
            document.title = changes["title"]

        self.audit.record(
            action="document.updated",
            resource_type="document",
            resource_id=document.id,
            workspace_id=workspace_id,
            actor_id=actor.id,
            **{k: str(v) for k, v in changes.items()},
        )
        await self.session.commit()
        await self.session.refresh(document)
        return document

    async def trash(self, actor: User, workspace_id: uuid.UUID, document_id: uuid.UUID) -> None:
        """Soft delete: hide the document but keep row + object for restore."""
        document = await self.get(workspace_id, document_id)
        document.deleted_at = datetime.now(UTC)
        self.audit.record(
            action="document.trashed",
            resource_type="document",
            resource_id=document.id,
            workspace_id=workspace_id,
            actor_id=actor.id,
            file_name=document.file_name,
        )
        await self.session.commit()

    async def restore(
        self, actor: User, workspace_id: uuid.UUID, document_id: uuid.UUID
    ) -> Document:
        document = await self._get_any(workspace_id, document_id)
        if document.deleted_at is None:
            raise NotFoundError("document is not in the trash")
        document.deleted_at = None
        self.audit.record(
            action="document.restored",
            resource_type="document",
            resource_id=document.id,
            workspace_id=workspace_id,
            actor_id=actor.id,
            file_name=document.file_name,
        )
        await self.session.commit()
        await self.session.refresh(document)
        return document

    async def list_trash(self, workspace_id: uuid.UUID) -> list[Document]:
        return await self.repository.list_trashed(workspace_id)

    async def delete_permanently(
        self, actor: User, workspace_id: uuid.UUID, document_id: uuid.UUID
    ) -> None:
        """Remove the row, then the object. DB first: a failed object delete
        leaves a harmless orphan, never a row pointing at a missing file."""
        document = await self._get_any(workspace_id, document_id)
        key = document.storage_key
        self.audit.record(
            action="document.deleted",
            resource_type="document",
            resource_id=document.id,
            workspace_id=workspace_id,
            actor_id=actor.id,
            file_name=document.file_name,
        )
        await self.repository.delete(document)
        await self.session.commit()
        await self.storage.delete(key)

    async def _read_capped(self, upload: UploadFile) -> tuple[bytes, int, str]:
        """Read the upload in chunks, enforcing the size cap and hashing as we go."""
        cap = self.settings.max_upload_size_bytes
        digest = hashlib.sha256()
        buffer = bytearray()
        size = 0
        while chunk := await upload.read(CHUNK_SIZE):
            size += len(chunk)
            if size > cap:
                raise PayloadTooLargeError(f"file exceeds the {cap}-byte upload limit")
            digest.update(chunk)
            buffer.extend(chunk)
        return bytes(buffer), size, digest.hexdigest()

    async def _get_any(self, workspace_id: uuid.UUID, document_id: uuid.UUID) -> Document:
        """Fetch active OR trashed (restore/permanent delete need trashed rows)."""
        document = await self.repository.get(document_id)
        if document is None or document.workspace_id != workspace_id:
            raise NotFoundError("document not found")
        return document

    async def _require_folder(self, workspace_id: uuid.UUID, folder_id: uuid.UUID) -> None:
        folder = await self.folders.get(folder_id)
        if folder is None or folder.workspace_id != workspace_id:
            raise NotFoundError("folder not found")
