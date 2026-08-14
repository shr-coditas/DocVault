"""Turn retrieved chunks into a cited answer.

This is the last step of the pipeline and the first one that sends workspace
document text to a third party. Everything before it - the permission predicate
inside the vector query, the intent gate, the guardrail chain - exists partly so
that what reaches this point is already the smallest, most legitimate set of
passages the asking user is allowed to see.

Three jobs, in order:

1. **Budget.** Take sources best-first until the token budget or the source cap
   is reached. Retrieval already ranked them; this only decides how many fit.
2. **Generate.** One call, through the ``ChatModel`` seam.
3. **Resolve citations.** Parse the bracketed markers the model wrote and map
   them back to the sources actually supplied, dropping any that do not resolve.

What it deliberately does *not* do is retry, re-rank, or re-query. A grade-and-
rewrite loop belongs in the graph-based slice where it can be measured; adding
an unmeasured retry here would just multiply cost.
"""

import re
from collections.abc import Sequence

import structlog

from app.ai import prompts
from app.config import Settings, get_settings
from app.services.ai_types import (
    AnswerAttempt,
    Citation,
    GeneratedAnswer,
    GenerationFailure,
    SearchHit,
)
from app.services.llm_service import ChatModel, LLMUnavailableError
from app.services.token_counting import (
    ConservativeGenerationTokenCounter,
    GenerationTokenCounter,
)

logger = structlog.stdlib.get_logger("docvault.answer")

# [1], [2, 3], [4,5] - one or more numbers in a single bracket. Matches what the
# system prompt asks for; anything else the model writes is simply not a citation
# as far as this parser is concerned, which is the safe direction to fail.
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class AnswerService:
    def __init__(
        self,
        model: ChatModel,
        settings: Settings | None = None,
        token_counter: GenerationTokenCounter | None = None,
    ) -> None:
        self.model = model
        self.settings = settings or get_settings()
        self.tokens = token_counter or ConservativeGenerationTokenCounter()

    async def answer(self, question: str, hits: Sequence[SearchHit]) -> AnswerAttempt:
        """Generate a cited answer and report exactly which sources were supplied.

        Failure remains a value rather than an exception: a provider outage must
        not discard successful retrieval, and the conversation ledger still
        needs to record which passages were sent before the call failed.
        """
        sources = tuple(self.select_sources(hits, question))
        if not sources:
            # Nothing grounded to say. Calling the model here would invite it to
            # answer from its own knowledge - the one thing the prompt forbids -
            # and bill us for the privilege.
            return AnswerAttempt(None, failure=GenerationFailure.NO_SOURCES)

        system = prompts.SYSTEM_PROMPT
        user = prompts.build_user_prompt(question, sources)

        try:
            completion = await self.model.complete(system, user)
        except LLMUnavailableError as exc:
            # Logged as a fact about the provider, not as an application error:
            # the request still succeeds, with sources and no answer.
            logger.warning(
                "generation_unavailable",
                failure_type=type(exc).__name__,
                sources=len(sources),
            )
            return AnswerAttempt(
                None,
                selected_sources=sources,
                failure=GenerationFailure.PROVIDER_UNAVAILABLE,
            )

        citations = self.resolve_citations(completion.text, sources)
        logger.info(
            "answer_generated",
            model=completion.model,
            sources=len(sources),
            citations=len(citations),
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
        )
        return AnswerAttempt(
            GeneratedAnswer(
                text=completion.text.strip(),
                model=completion.model,
                citations=citations,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
            ),
            selected_sources=sources,
        )

    def select_sources(self, hits: Sequence[SearchHit], question: str = "") -> list[SearchHit]:
        """Take hits best-first until the budget or the source cap runs out.

        Order is preserved rather than re-sorted: retrieval already ranked these
        by relevance, and the model reads earlier sources as more salient.

        A single chunk larger than the whole budget is still included when it is
        the first one - a truncated-but-grounded answer beats refusing to answer
        because one document chunks badly.
        """
        fixed = self.tokens.count_tokens(prompts.SYSTEM_PROMPT) + self.tokens.count_tokens(question)
        window_budget = max(
            0,
            self.settings.llm_context_window_tokens
            - self.settings.llm_reserved_output_tokens
            - fixed,
        )
        budget = min(self.settings.answer_context_token_budget, window_budget)
        selected: list[SearchHit] = []
        spent = 0
        for hit in hits:
            if len(selected) >= self.settings.answer_max_sources:
                break
            rendered_tokens = self.tokens.count_tokens(prompts.format_sources([hit]))
            if selected and spent + rendered_tokens > budget:
                # keep going: a later, smaller chunk may still fit
                continue
            selected.append(hit)
            spent += rendered_tokens
        return selected

    @staticmethod
    def resolve_citations(text: str, sources: Sequence[SearchHit]) -> tuple[Citation, ...]:
        """Map the markers the model wrote back to the sources it was given.

        Two rules, both about not over-claiming:

        - a marker outside ``1..len(sources)`` is **dropped**. The model
          occasionally invents one, and a citation that resolves to nothing is
          worse than no citation - it looks authoritative and cannot be checked.
        - each source is reported once even if cited repeatedly, in the order the
          model first used it, so the list reads like a bibliography rather than
          a log.
        """
        seen: set[int] = set()
        citations: list[Citation] = []
        for match in _CITATION_RE.finditer(text):
            for part in match.group(1).split(","):
                marker = int(part.strip())
                if marker in seen or not 1 <= marker <= len(sources):
                    continue
                seen.add(marker)
                hit = sources[marker - 1]
                citations.append(
                    Citation(
                        marker=marker,
                        document_id=hit.document_id,
                        document_title=hit.document_title,
                        chunk_id=hit.chunk_id,
                        heading=hit.heading,
                        source_spans=hit.source_spans,
                        index_generation=hit.index_generation,
                        logical_key=hit.logical_key,
                    )
                )
        return tuple(citations)
