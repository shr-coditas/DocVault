from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Supervision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["search", "answer", "clarify", "refuse", "unsupported"]
    question: str = Field(default="", max_length=4000)
    searches: list[str] = Field(default_factory=list, max_length=3)
    reason: str = Field(default="unspecified", pattern=r"^[a-z][a-z0-9_]{0,60}$")
