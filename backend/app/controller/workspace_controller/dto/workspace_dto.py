import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

RoleName = Literal["owner", "editor", "viewer"]


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)


class WorkspaceUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=500)


class WorkspaceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str
    created_by: uuid.UUID
    created_at: datetime


class WorkspaceWithRoleOut(WorkspaceOut):
    my_role: str


class MemberOut(BaseModel):
    user_id: uuid.UUID
    email: str
    full_name: str
    role: str
    joined_at: datetime


class AddMemberRequest(BaseModel):
    email: EmailStr
    role: RoleName = "viewer"


class UpdateMemberRequest(BaseModel):
    role: RoleName
