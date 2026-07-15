import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.controller.workspace_controller.dto.workspace_dto import (
    AddMemberRequest,
    MemberOut,
    UpdateMemberRequest,
    WorkspaceCreate,
    WorkspaceUpdate,
)
from app.exceptions import ConflictError, NotFoundError
from app.models.user import User
from app.models.workspace import Workspace, WorkspaceMember
from app.repository.rbac_repository import RbacRepository
from app.repository.user_repository import UserRepository
from app.repository.workspace_repository import WorkspaceRepository
from app.services.audit_service import AuditService
from app.services.permission_service import PermissionService
from app.utils.rbac_catalog import OWNER


class WorkspaceService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = WorkspaceRepository(session)
        self.users = UserRepository(session)
        self.rbac = RbacRepository(session)
        self.audit = AuditService(session)
        self.permissions = PermissionService(session)

    async def create(self, actor: User, data: WorkspaceCreate) -> Workspace:
        workspace = Workspace(name=data.name, description=data.description, created_by=actor.id)
        self.repository.add(workspace)
        await self.session.flush()  # materialize workspace.id

        owner_role = await self.permissions.role_by_name(OWNER)
        self.repository.add_member(
            WorkspaceMember(workspace_id=workspace.id, user_id=actor.id, role_id=owner_role.id)
        )
        self.audit.record(
            action="workspace.created",
            resource_type="workspace",
            resource_id=workspace.id,
            workspace_id=workspace.id,
            actor_id=actor.id,
            name=workspace.name,
        )
        await self.session.commit()
        await self.session.refresh(workspace)
        return workspace

    async def list_for_user(self, user_id: uuid.UUID) -> list[tuple[Workspace, str]]:
        return await self.repository.list_for_user(user_id)

    async def get(self, workspace_id: uuid.UUID) -> Workspace:
        workspace = await self.repository.get(workspace_id)
        if workspace is None:
            raise NotFoundError("workspace not found")
        return workspace

    async def update(
        self, actor: User, workspace_id: uuid.UUID, data: WorkspaceUpdate
    ) -> Workspace:
        workspace = await self.get(workspace_id)
        changes = data.model_dump(exclude_unset=True, exclude_none=True)
        for field, value in changes.items():
            setattr(workspace, field, value)
        self.audit.record(
            action="workspace.updated",
            resource_type="workspace",
            resource_id=workspace.id,
            workspace_id=workspace.id,
            actor_id=actor.id,
            **changes,
        )
        await self.session.commit()
        await self.session.refresh(workspace)
        return workspace

    async def delete(self, actor: User, workspace_id: uuid.UUID) -> None:
        workspace = await self.get(workspace_id)
        self.audit.record(
            action="workspace.deleted",
            resource_type="workspace",
            resource_id=workspace.id,
            workspace_id=workspace.id,
            actor_id=actor.id,
            name=workspace.name,
        )
        await self.repository.delete(workspace)
        await self.session.commit()

    async def members(self, workspace_id: uuid.UUID) -> list[MemberOut]:
        return [
            MemberOut(
                user_id=user.id,
                email=user.email,
                full_name=user.full_name,
                role=role_name,
                joined_at=member.joined_at,
            )
            for member, user, role_name in await self.repository.members_with_roles(workspace_id)
        ]

    async def add_member(
        self, actor: User, workspace_id: uuid.UUID, data: AddMemberRequest
    ) -> MemberOut:
        user = await self.users.get_by_email(data.email.lower())
        if user is None:
            raise NotFoundError("no user with this email")
        if await self.repository.get_member(workspace_id, user.id) is not None:
            raise ConflictError("user is already a member of this workspace")

        role = await self.permissions.role_by_name(data.role)
        member = WorkspaceMember(workspace_id=workspace_id, user_id=user.id, role_id=role.id)
        self.repository.add_member(member)
        self.audit.record(
            action="member.added",
            resource_type="workspace_member",
            resource_id=user.id,
            workspace_id=workspace_id,
            actor_id=actor.id,
            email=user.email,
            role=data.role,
        )
        await self.session.commit()
        await self.session.refresh(member)
        return MemberOut(
            user_id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=data.role,
            joined_at=member.joined_at,
        )

    async def update_member_role(
        self, actor: User, workspace_id: uuid.UUID, user_id: uuid.UUID, data: UpdateMemberRequest
    ) -> None:
        member = await self.repository.get_member(workspace_id, user_id)
        if member is None:
            raise NotFoundError("member not found")

        current_role = await self.rbac.get_role(member.role_id)
        assert current_role is not None
        if (
            current_role.name == OWNER
            and data.role != OWNER
            and await self.repository.owner_count(workspace_id, OWNER) == 1
        ):
            raise ConflictError("workspace must keep at least one owner")

        member.role_id = (await self.permissions.role_by_name(data.role)).id
        self.audit.record(
            action="member.role_changed",
            resource_type="workspace_member",
            resource_id=user_id,
            workspace_id=workspace_id,
            actor_id=actor.id,
            from_role=current_role.name,
            to_role=data.role,
        )
        await self.session.commit()

    async def remove_member(self, actor: User, workspace_id: uuid.UUID, user_id: uuid.UUID) -> None:
        member = await self.repository.get_member(workspace_id, user_id)
        if member is None:
            raise NotFoundError("member not found")

        role = await self.rbac.get_role(member.role_id)
        assert role is not None
        if role.name == OWNER and await self.repository.owner_count(workspace_id, OWNER) == 1:
            raise ConflictError("workspace must keep at least one owner")

        await self.repository.delete_member(member)
        self.audit.record(
            action="member.removed",
            resource_type="workspace_member",
            resource_id=user_id,
            workspace_id=workspace_id,
            actor_id=actor.id,
        )
        await self.session.commit()
