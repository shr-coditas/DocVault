"""Provider-neutral structured-output model seam for bounded graph decisions."""

import asyncio
from functools import lru_cache
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings

StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


class StructuredModelUnavailableError(RuntimeError):
    """A structured model call failed or returned an invalid contract."""


class StructuredModel(Protocol):
    model_name: str

    async def complete(
        self,
        system: str,
        user: str,
        schema: type[StructuredOutput],
        *,
        timeout_seconds: float,
    ) -> StructuredOutput: ...


class LangChainStructuredModel:
    """Lazy provider client shared by analysis, planning, and grading.

    LangChain and provider response types stop here. Services receive validated
    Pydantic models through the narrow ``StructuredModel`` protocol, and tests
    replace this class with a deterministic in-process fake.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.provider = self.settings.agent_model_provider or self.settings.llm_provider
        self.model_name = self.settings.agent_model or self.settings.llm_model
        self._models: dict[type[BaseModel], Any] = {}

    def _client(self, schema: type[StructuredOutput]) -> Any:
        model = self._models.get(schema)
        if model is None:
            from langchain.chat_models import init_chat_model

            credentials: dict[str, Any] = {}
            if self.provider == "google_genai" and self.settings.google_studio_api_key:
                credentials["api_key"] = self.settings.google_studio_api_key
            if self.provider == "groq" and self.settings.groq_api_key:
                credentials["api_key"] = self.settings.groq_api_key
            base_model = init_chat_model(
                model=self.model_name,
                model_provider=self.provider,
                temperature=0,
                max_tokens=self.settings.agent_structured_max_output_tokens,
                **credentials,
            )
            model = base_model.with_structured_output(schema)
            self._models[schema] = model
        return model

    async def complete(
        self,
        system: str,
        user: str,
        schema: type[StructuredOutput],
        *,
        timeout_seconds: float,
    ) -> StructuredOutput:
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            response = await asyncio.wait_for(
                self._client(schema).ainvoke(
                    [SystemMessage(content=system), HumanMessage(content=user)]
                ),
                timeout=timeout_seconds,
            )
            return response if isinstance(response, schema) else schema.model_validate(response)
        except TimeoutError as exc:
            raise StructuredModelUnavailableError("structured model timed out") from exc
        except ValidationError as exc:
            raise StructuredModelUnavailableError(
                "structured model returned an invalid result"
            ) from exc
        except Exception as exc:
            raise StructuredModelUnavailableError(
                f"{type(exc).__name__} from the structured model provider"
            ) from exc


@lru_cache(maxsize=1)
def get_default_structured_model() -> LangChainStructuredModel:
    return LangChainStructuredModel()
