import hashlib
import mimetypes
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import PurePosixPath

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from uuid6 import uuid7

from app.config import get_settings
from app.controller.document_controller.dto.document_dto import DocumentUpdate
from app.exceptions import NotFoundError, PayloadTooLargeError, UnsupportedMediaTypeError
from app.models.document import Document, DocumentVisibility
from app.models.document_grant import DocumentGrant, PrincipalType
from app.models.user import User
from app.repository.document_grant_repository import DocumentGrantRepository
from app.repository.document_repository import DocumentRepository
from app.repository.folder_repository import FolderRepository
from app.repository.team_repository import TeamRepository
from app.services.audit_service import AuditService
from app.services.permission_service import PermissionService
from app.services.storage_service import CHUNK_SIZE, StorageService, document_key
from app.utils.rbac_catalog import OWNER

DEFAULT_MIME = "application/octet-stream"


class DocumentService:
    def __init__(self, session: AsyncSession, storage: StorageService) -> None:
        self.session = session
        self.storage = storage
        self.repository = DocumentRepository(session)
        self.folders = FolderRepository(session)
        self.grants = DocumentGrantRepository(session)
        self.teams = TeamRepository(session)
        self.permissions = PermissionService(session)
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
        self._require_supported_type(file_name)
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

    async def get(self, actor: User, workspace_id: uuid.UUID, document_id: uuid.UUID) -> Document:
        """Active documents only — trashed ones are invisible here (404)."""
        document = await self._get_any(actor, workspace_id, document_id)
        if document.deleted_at is not None:
            raise NotFoundError("document not found")
        return document

    async def list_documents(
        self,
        actor: User,
        workspace_id: uuid.UUID,
        folder_id: uuid.UUID | None,
        *,
        every_folder: bool = False,
    ) -> list[Document]:
        access = await self._access_filter(actor, workspace_id)
        return await self.repository.list_in_workspace(
            workspace_id, folder_id, access=access, every_folder=every_folder
        )

    async def open_stream(
        self, actor: User, workspace_id: uuid.UUID, document_id: uuid.UUID
    ) -> tuple[AsyncIterator[bytes], str, str]:
        document = await self.get(actor, workspace_id, document_id)
        return self.storage.stream(document.storage_key), document.file_name, document.mime_type

    async def update(
        self, actor: User, workspace_id: uuid.UUID, document_id: uuid.UUID, data: DocumentUpdate
    ) -> Document:
        """Rename (title) and/or move (folder_id; None = workspace root)."""
        document = await self.get(actor, workspace_id, document_id)
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
        document = await self.get(actor, workspace_id, document_id)
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
        document = await self._get_any(actor, workspace_id, document_id)
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

    async def list_trash(self, actor: User, workspace_id: uuid.UUID) -> list[Document]:
        access = await self._access_filter(actor, workspace_id)
        return await self.repository.list_trashed(workspace_id, access=access)

    async def delete_permanently(
        self, actor: User, workspace_id: uuid.UUID, document_id: uuid.UUID
    ) -> None:
        """Remove the row, then the object. DB first: a failed object delete
        leaves a harmless orphan, never a row pointing at a missing file."""
        document = await self._get_any(actor, workspace_id, document_id)
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

    # -- sharing -----------------------------------------------------------

    async def set_visibility(
        self,
        actor: User,
        workspace_id: uuid.UUID,
        document_id: uuid.UUID,
        visibility: DocumentVisibility,
    ) -> Document:
        document = await self.get(actor, workspace_id, document_id)
        document.visibility = visibility
        self.audit.record(
            action="document.visibility_changed",
            resource_type="document",
            resource_id=document.id,
            workspace_id=workspace_id,
            actor_id=actor.id,
            visibility=visibility.value,
        )
        await self.session.commit()
        await self.session.refresh(document)
        return document

    async def list_grants(
        self, actor: User, workspace_id: uuid.UUID, document_id: uuid.UUID
    ) -> list[DocumentGrant]:
        await self.get(actor, workspace_id, document_id)
        return await self.grants.list_for_document(document_id)

    async def add_grant(
        self,
        actor: User,
        workspace_id: uuid.UUID,
        document_id: uuid.UUID,
        principal_type: PrincipalType,
        principal_id: uuid.UUID,
    ) -> DocumentGrant:
        await self.get(actor, workspace_id, document_id)
        await self._require_principal(workspace_id, principal_type, principal_id)

        existing = await self.grants.get(document_id, principal_type, principal_id)
        if existing is not None:  # idempotent: sharing twice is a no-op
            return existing

        grant = DocumentGrant(
            document_id=document_id,
            principal_type=principal_type,
            principal_id=principal_id,
            granted_by=actor.id,
        )
        self.grants.add(grant)
        self.audit.record(
            action="document.grant_added",
            resource_type="document",
            resource_id=document_id,
            workspace_id=workspace_id,
            actor_id=actor.id,
            principal_type=principal_type.value,
            principal_id=str(principal_id),
        )
        await self.session.commit()
        await self.session.refresh(grant)
        return grant

    async def remove_grant(
        self,
        actor: User,
        workspace_id: uuid.UUID,
        document_id: uuid.UUID,
        principal_type: PrincipalType,
        principal_id: uuid.UUID,
    ) -> None:
        await self.get(actor, workspace_id, document_id)
        grant = await self.grants.get(document_id, principal_type, principal_id)
        if grant is None:
            raise NotFoundError("grant not found")
        await self.grants.delete(grant)
        self.audit.record(
            action="document.grant_removed",
            resource_type="document",
            resource_id=document_id,
            workspace_id=workspace_id,
            actor_id=actor.id,
            principal_type=principal_type.value,
            principal_id=str(principal_id),
        )
        await self.session.commit()

    # -- internals ---------------------------------------------------------

    def _require_supported_type(self, file_name: str) -> None:
        """Reject anything the ingestion pipeline could not read.

        Extension-based, and checked before the body is read so an unsupported
        file is refused without buffering it. This is a scope control, not a
        security boundary: an attacker can rename a file, so content sniffing
        and scanning still belong on the roadmap.
        """
        allowed = self.settings.allowed_upload_extensions
        if PurePosixPath(file_name).suffix.lower() not in allowed:
            raise UnsupportedMediaTypeError(
                f"file type not accepted; allowed: {', '.join(allowed)}"
            )

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

    async def _get_any(
        self, actor: User, workspace_id: uuid.UUID, document_id: uuid.UUID
    ) -> Document:
        """Fetch active OR trashed (restore/permanent delete need trashed rows),
        then enforce per-document visibility (invisible → 404)."""
        document = await self.repository.get(document_id)
        if document is None or document.workspace_id != workspace_id:
            raise NotFoundError("document not found")
        await self._ensure_can_access(actor, document)
        return document

    async def _ensure_can_access(self, actor: User, document: Document) -> None:
        """Visibility gate layered on top of the router's workspace-role check.

        ``workspace`` visibility is open to the whole workspace; ``private`` and
        ``team`` both mean *restricted*, and are opened up by grants. A grant
        counts the same whether it names a user or a team — visibility decides
        whether grants are consulted, never which kind of grant is honoured.
        """
        if document.visibility == DocumentVisibility.WORKSPACE:
            return
        if document.owner_id == actor.id:
            return
        role = await self.permissions.workspace_role_name(actor.id, document.workspace_id)
        if role == OWNER:  # workspace owner sees everything (admin override)
            return
        if await self.grants.user_has_grant(document.id, actor.id):
            return
        team_ids = await self.teams.team_ids_for_user(document.workspace_id, actor.id)
        if await self.grants.team_grant_exists(document.id, team_ids):
            return
        raise NotFoundError("document not found")

    async def _access_filter(
        self, actor: User, workspace_id: uuid.UUID
    ) -> tuple[uuid.UUID, list[uuid.UUID]] | None:
        """Access params for list queries; None = admin (no visibility filter)."""
        role = await self.permissions.workspace_role_name(actor.id, workspace_id)
        if role == OWNER:
            return None
        team_ids = await self.teams.team_ids_for_user(workspace_id, actor.id)
        return actor.id, team_ids

    async def _require_principal(
        self, workspace_id: uuid.UUID, principal_type: PrincipalType, principal_id: uuid.UUID
    ) -> None:
        """A grant target must live in this workspace."""
        if principal_type == PrincipalType.USER:
            role = await self.permissions.workspace_role_name(principal_id, workspace_id)
            if role is None:
                raise NotFoundError("user is not a member of this workspace")
        else:
            team = await self.teams.get(principal_id)
            if team is None or team.workspace_id != workspace_id:
                raise NotFoundError("team not found")

    async def _require_folder(self, workspace_id: uuid.UUID, folder_id: uuid.UUID) -> None:
        folder = await self.folders.get(folder_id)
        if folder is None or folder.workspace_id != workspace_id:
            raise NotFoundError("folder not found")
