"""AnswerService in isolation: budgeting, citation resolution, failure.

No database and no HTTP - these are the decisions the service makes on its own,
and they are the ones that are painful to pin down through a full request. The
end-to-end path is covered in ``test_query.py``.
"""

import re
import uuid

import pytest

from app.config import Settings
from app.services.ai_types import ChunkSourceSpan, SearchHit, SourceLocation
from app.services.answer_service import AnswerService
from tests.fakes import FakeChatModel, UnavailableChatModel


class TestGenerationCounter:
    """Treat the fixture's explicit token marker as the rendered prompt cost."""

    def count_tokens(self, text: str) -> int:
        match = re.search(r"\[tokens=(\d+)\]", text)
        return int(match.group(1)) if match else 0


def hit(marker: str, *, tokens: int = 10, page: int | None = None) -> SearchHit:
    return SearchHit(
        document_id=uuid.uuid4(),
        document_title=f"{marker}.txt",
        file_name=f"{marker}.txt",
        chunk_id=uuid.uuid4(),
        chunk_index=0,
        content=f"content of {marker} [tokens={tokens}]",
        score=0.9,
        token_count=tokens,
        source_spans=(
            ChunkSourceSpan(
                0,
                0,
                len(f"content of {marker} [tokens={tokens}]"),
                SourceLocation(page_number=page),
            ),
        )
        if page is not None
        else (),
    )


# -- budgeting -------------------------------------------------------------


def test_sources_are_taken_best_first_up_to_the_cap() -> None:
    service = AnswerService(
        FakeChatModel(), Settings(answer_max_sources=3), TestGenerationCounter()
    )
    hits = [hit(str(i)) for i in range(10)]

    selected = service.select_sources(hits)

    # order preserved: retrieval already ranked these, and the model reads
    # earlier sources as more salient
    assert [source.document_title for source in selected] == ["0.txt", "1.txt", "2.txt"]


def test_the_token_budget_stops_selection() -> None:
    service = AnswerService(
        FakeChatModel(),
        Settings(answer_context_token_budget=25, answer_max_sources=99),
        TestGenerationCounter(),
    )

    selected = service.select_sources(
        [hit("a", tokens=10), hit("b", tokens=10), hit("c", tokens=10)]
    )

    assert [source.document_title for source in selected] == ["a.txt", "b.txt"]


def test_a_later_smaller_chunk_still_fits_after_one_is_skipped() -> None:
    """Skipping an over-budget chunk must not end selection early."""
    service = AnswerService(
        FakeChatModel(), Settings(answer_context_token_budget=30), TestGenerationCounter()
    )

    selected = service.select_sources(
        [hit("a", tokens=20), hit("big", tokens=500), hit("c", tokens=5)]
    )

    assert [source.document_title for source in selected] == ["a.txt", "c.txt"]


def test_a_single_oversized_chunk_is_still_answered_over() -> None:
    """Truncated-but-grounded beats refusing because one document chunks badly."""
    service = AnswerService(
        FakeChatModel(), Settings(answer_context_token_budget=10), TestGenerationCounter()
    )

    selected = service.select_sources([hit("huge", tokens=5000)])

    assert [source.document_title for source in selected] == ["huge.txt"]


async def test_no_hits_means_no_model_call() -> None:
    model = FakeChatModel()
    service = AnswerService(model)

    attempt = await service.answer("anything?", [])

    assert attempt.answer is None
    assert attempt.selected_sources == ()
    assert attempt.failure == "no_sources"
    assert model.calls == 0


# -- citation resolution ---------------------------------------------------


def test_markers_resolve_to_the_sources_supplied() -> None:
    sources = [hit("a", page=3), hit("b")]

    citations = AnswerService.resolve_citations("Alpha [1]. Beta [2].", sources)

    assert [citation.marker for citation in citations] == [1, 2]
    assert citations[0].document_title == "a.txt"
    assert citations[0].page_number == 3
    assert citations[1].chunk_id == sources[1].chunk_id


def test_a_multi_number_marker_resolves_to_each_source() -> None:
    citations = AnswerService.resolve_citations("Both agree [1, 2].", [hit("a"), hit("b")])

    assert [citation.marker for citation in citations] == [1, 2]


@pytest.mark.parametrize("text", ["Out of range [3].", "Zero [0].", "Way off [99]."])
def test_unresolvable_markers_are_dropped(text: str) -> None:
    """A citation that resolves to nothing looks authoritative and cannot be checked."""
    assert AnswerService.resolve_citations(text, [hit("a"), hit("b")]) == ()


def test_a_source_cited_repeatedly_is_reported_once_in_first_use_order() -> None:
    citations = AnswerService.resolve_citations(
        "[2] then [1] then [2] again.", [hit("a"), hit("b")]
    )

    assert [citation.marker for citation in citations] == [2, 1]


def test_prose_without_markers_yields_no_citations() -> None:
    assert AnswerService.resolve_citations("The documents do not cover this.", [hit("a")]) == ()


# -- failure ---------------------------------------------------------------


async def test_a_provider_failure_returns_none_rather_than_raising() -> None:
    """A failed generation must not discard a successful retrieval."""
    model = UnavailableChatModel()
    service = AnswerService(model)

    attempt = await service.answer("what is the policy?", [hit("a")])

    assert attempt.answer is None
    assert len(attempt.selected_sources) == 1
    assert attempt.failure == "provider_unavailable"
    assert model.calls == 1


async def test_a_successful_answer_carries_the_model_and_its_token_counts() -> None:
    service = AnswerService(FakeChatModel(reply="Grounded [1]."))

    attempt = await service.answer("what is the policy?", [hit("a")])
    answer = attempt.answer

    assert answer is not None
    assert answer.text == "Grounded [1]."
    assert answer.model == "fake-chat"
    assert (answer.input_tokens, answer.output_tokens) == (11, 7)
    assert [citation.marker for citation in answer.citations] == [1]
    assert [source.document_title for source in attempt.selected_sources] == ["a.txt"]
