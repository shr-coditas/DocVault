import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TeamCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class TeamOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    created_at: datetime


class AddTeamMemberRequest(BaseModel):
    user_id: uuid.UUID


class TeamMemberOut(BaseModel):
    user_id: uuid.UUID
    email: str
    full_name: str
