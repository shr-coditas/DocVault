from pydantic import BaseModel, ConfigDict, Field


class ScopeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    in_scope: bool
    reason: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$",
    )
