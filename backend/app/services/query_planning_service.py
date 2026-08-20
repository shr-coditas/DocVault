"""Bounded retrieval planning after safe contextual query resolution."""

import json
from functools import lru_cache
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import Settings, get_settings
from app.services.ai_types import QueryAnalysis, QueryPlan, QueryTask
from app.services.structured_model_service import (
    StructuredModel,
    StructuredModelUnavailableError,
    get_default_structured_model,
)

SYSTEM_PROMPT = """Plan searches for one authorized DocVault question.

The application, not you, owns the actor, workspace and document scope. Produce
only search wording and required answer aspects; never add identifiers, filters,
tools, permissions or a broader scope. Use at most three concise searches. Ask a
clarifying question instead of searching when required information can only come
from the user. Return only the requested structured result."""


class QueryPlanOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: QueryTask
    search_queries: list[str] = Field(default_factory=list, max_length=3)
    required_aspects: list[str] = Field(default_factory=list, max_length=8)
    needs_clarification: bool = False
    clarification_question: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_route(self) -> "QueryPlanOutput":
        self.search_queries = list(
            dict.fromkeys(query.strip() for query in self.search_queries if query.strip())
        )
        self.required_aspects = list(
            dict.fromkeys(aspect.strip() for aspect in self.required_aspects if aspect.strip())
        )
        if self.needs_clarification:
            if self.search_queries or not self.clarification_question:
                raise ValueError("clarification plans require a question and no searches")
        elif not self.search_queries or self.clarification_question is not None:
            raise ValueError("retrieval plans require searches and no clarification question")
        if any(len(query) > 4000 for query in self.search_queries):
            raise ValueError("a planned search exceeds the query limit")
        return self


class QueryPlanner(Protocol):
    async def plan(self, query: str, analysis: QueryAnalysis) -> QueryPlan: ...


class DirectQueryPlanner:
    async def plan(self, query: str, analysis: QueryAnalysis) -> QueryPlan:
        if analysis.task is QueryTask.AMBIGUOUS:
            return QueryPlan(
                task=analysis.task,
                search_queries=(),
                needs_clarification=True,
                clarification_question="What document, topic, or earlier answer should I use?",
            )
        return QueryPlan(task=analysis.task, search_queries=(query.strip(),))


class StructuredQueryPlanner:
    def __init__(self, model: StructuredModel, settings: Settings | None = None) -> None:
        self.model = model
        self.settings = settings or get_settings()

    async def plan(self, query: str, analysis: QueryAnalysis) -> QueryPlan:
        output = await self.model.complete(
            SYSTEM_PROMPT,
            "Plan searches from this JSON data:\n"
            + json.dumps(
                {"query": query, "task": analysis.task.value},
                ensure_ascii=False,
            ),
            QueryPlanOutput,
            timeout_seconds=self.settings.agent_planning_timeout_seconds,
        )
        queries = tuple(output.search_queries[: self.settings.agent_max_subqueries])
        return QueryPlan(
            task=output.task,
            search_queries=queries,
            required_aspects=tuple(output.required_aspects),
            needs_clarification=output.needs_clarification,
            clarification_question=output.clarification_question,
        )


class FallbackQueryPlanner:
    def __init__(self, primary: QueryPlanner, fallback: QueryPlanner | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or DirectQueryPlanner()

    async def plan(self, query: str, analysis: QueryAnalysis) -> QueryPlan:
        try:
            return await self.primary.plan(query, analysis)
        except StructuredModelUnavailableError:
            return await self.fallback.plan(query, analysis)


@lru_cache(maxsize=1)
def get_default_query_planner() -> FallbackQueryPlanner:
    return FallbackQueryPlanner(StructuredQueryPlanner(get_default_structured_model()))
