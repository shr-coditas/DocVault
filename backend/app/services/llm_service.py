"""The chat-model seam. One narrow Protocol, one LangChain-backed implementation.

Every LLM call in this repository goes through ``init_chat_model``, so the
provider is an environment variable rather than an import. Groq is the default
because its free tier makes the whole pipeline runnable without a card; Claude
is the intended choice for generation quality and for the LLM-as-judge work
later. Neither name appears anywhere but config.

**The Protocol is deliberately narrower than LangChain's.** ``complete`` takes
two strings and returns one, and nothing outside this module imports a LangChain
type. That is what makes the test double three lines long instead of a mock of
someone else's class hierarchy - the same trade the ``Embedder`` seam makes, for
the same reason.

Two things this module owns that a caller must not be able to get wrong:

- **the timeout**, enforced with ``asyncio.wait_for``. Every provider spells its
  own timeout argument differently (Groq calls it ``request_timeout``), so a
  provider argument would leak the provider back into config. A wall-clock guard
  around the call is provider-neutral and covers connect, generate, and read.
- **failure as a value.** A provider outage, a rate limit, or a missing API key
  raises here and is caught by the caller, which degrades to returning the
  retrieved sources without an answer. A 500 would throw away a perfectly good
  retrieval because generation was unavailable.
"""

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

from app.config import Settings, get_settings


class LLMUnavailableError(RuntimeError):
    """The model could not be reached, timed out, or refused the request.

    Deliberately one exception for every provider failure. The caller's response
    is the same in all cases - return the sources, skip the answer - and a
    taxonomy nobody branches on is a taxonomy that goes stale.
    """


@dataclass(frozen=True, slots=True)
class Completion:
    """One model response, with what it cost.

    Token counts come from the provider and are ``None`` when it does not report
    them - which is why they are optional rather than zero. Zero would be a
    measurement; ``None`` is the absence of one, and the difference matters as
    soon as anyone totals these up.
    """

    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class ChatModel(Protocol):
    """Structural, so the test double needs no inheritance and no LangChain."""

    model_name: str

    async def complete(self, system: str, user: str) -> Completion: ...


class LangChainChatModel:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_name = self.settings.llm_model
        self._model: Any | None = None

    def _client(self) -> Any:
        """Build the model on first use, not at construction.

        The API key is only needed once a request actually needs generating, so a
        deployment with no key still starts, still serves search, and still
        answers everything the pipeline can answer without a model. Failing at
        import time would take the whole application down over an optional
        feature.
        """
        if self._model is None:
            # imported lazily for the same reason: LangChain is a heavy import
            # and the test suite never touches this path
            from langchain.chat_models import init_chat_model

            credentials: dict[str, Any] = {}
            # if self.settings.groq_api_key:
            #     credentials["api_key"] = self.settings.groq_api_key
            if self.settings.google_studio_api_key:
                credentials["api_key"] = self.settings.google_studio_api_key
            self._model = init_chat_model(
                model=self.settings.llm_model,
                model_provider=self.settings.llm_provider,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_output_tokens,
                **credentials,
            )
        return self._model

    async def complete(self, system: str, user: str) -> Completion:
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            response = await asyncio.wait_for(
                self._client().ainvoke([SystemMessage(content=system), HumanMessage(content=user)]),
                timeout=self.settings.llm_timeout_seconds,
            )
        except TimeoutError as exc:
            raise LLMUnavailableError(
                f"the model did not respond within {self.settings.llm_timeout_seconds}s"
            ) from exc
        except Exception as exc:
            # Everything else - auth, rate limits, connection resets, provider
            # 5xx - is the same event to the caller. The type is logged; the
            # provider's message is not, because it can quote the prompt back.
            raise LLMUnavailableError(f"{type(exc).__name__} from the model provider") from exc

        usage = getattr(response, "usage_metadata", None) or {}
        return Completion(
            # content is a str for a plain text reply, but the type allows a list
            # of content blocks; str() keeps one shape reaching the caller
            # response.content if isinstance(response.content, str) else str(response.content)
            text=response.text,
            model=self.model_name,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )


@lru_cache(maxsize=1)
def get_default_chat_model() -> LangChainChatModel:
    """Process-wide singleton - the client holds a connection pool."""
    return LangChainChatModel()
