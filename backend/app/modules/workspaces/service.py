import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, NotFoundError
from app.modules.audit.service import AuditService
from app.modules.rbac.catalog import OWNER
from app.modules.rbac.models import Role
from app.modules.rbac.service import PermissionService
from app.modules.users.models import User
from app.modules.users.repository import UserRepository
from app.modules.workspaces.models import Workspace, WorkspaceMember
from app.modules.workspaces.schemas import (
    AddMemberRequest,
    MemberOut,
    UpdateMemberRequest,
    WorkspaceCreate,
    WorkspaceUpdate,
)


class WorkspaceService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.audit = AuditService(session)
        self.permissions = PermissionService(session)

    async def create(self, actor: User, data: WorkspaceCreate) -> Workspace:
        workspace = Workspace(name=data.name, description=data.description, created_by=actor.id)
        self.session.add(workspace)
        await self.session.flush()  # materialize workspace.id

        owner_role = await self.permissions.role_by_name(OWNER)
        self.session.add(
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
        stmt = (
            select(Workspace, Role.name)
            .join(WorkspaceMember, WorkspaceMember.workspace_id == Workspace.id)
            .join(Role, Role.id == WorkspaceMember.role_id)
            .where(WorkspaceMember.user_id == user_id)
            .order_by(Workspace.created_at.desc())
        )
        return [(ws, role) for ws, role in (await self.session.execute(stmt)).all()]

    async def get(self, workspace_id: uuid.UUID) -> Workspace:
        workspace = await self.session.get(Workspace, workspace_id)
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
        await self.session.delete(workspace)
        await self.session.commit()

    async def members(self, workspace_id: uuid.UUID) -> list[MemberOut]:
        stmt = (
            select(WorkspaceMember, User, Role.name)
            .join(User, User.id == WorkspaceMember.user_id)
            .join(Role, Role.id == WorkspaceMember.role_id)
            .where(WorkspaceMember.workspace_id == workspace_id)
            .order_by(WorkspaceMember.joined_at)
        )
        return [
            MemberOut(
                user_id=user.id,
                email=user.email,
                full_name=user.full_name,
                role=role_name,
                joined_at=member.joined_at,
            )
            for member, user, role_name in (await self.session.execute(stmt)).all()
        ]

    async def add_member(
        self, actor: User, workspace_id: uuid.UUID, data: AddMemberRequest
    ) -> MemberOut:
        user = await UserRepository(self.session).get_by_email(data.email.lower())
        if user is None:
            raise NotFoundError("no user with this email")
        if await self.session.get(WorkspaceMember, (workspace_id, user.id)) is not None:
            raise ConflictError("user is already a member of this workspace")

        role = await self.permissions.role_by_name(data.role)
        member = WorkspaceMember(workspace_id=workspace_id, user_id=user.id, role_id=role.id)
        self.session.add(member)
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
        member = await self.session.get(WorkspaceMember, (workspace_id, user_id))
        if member is None:
            raise NotFoundError("member not found")

        current_role = await self.session.get(Role, member.role_id)
        assert current_role is not None
        if (
            current_role.name == OWNER
            and data.role != OWNER
            and await self._owner_count(workspace_id) == 1
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
        member = await self.session.get(WorkspaceMember, (workspace_id, user_id))
        if member is None:
            raise NotFoundError("member not found")

        role = await self.session.get(Role, member.role_id)
        assert role is not None
        if role.name == OWNER and await self._owner_count(workspace_id) == 1:
            raise ConflictError("workspace must keep at least one owner")

        await self.session.delete(member)
        self.audit.record(
            action="member.removed",
            resource_type="workspace_member",
            resource_id=user_id,
            workspace_id=workspace_id,
            actor_id=actor.id,
        )
        await self.session.commit()

    async def _owner_count(self, workspace_id: uuid.UUID) -> int:
        stmt = (
            select(func.count())
            .select_from(WorkspaceMember)
            .join(Role, Role.id == WorkspaceMember.role_id)
            .where(WorkspaceMember.workspace_id == workspace_id, Role.name == OWNER)
        )
        return (await self.session.execute(stmt)).scalar_one()
