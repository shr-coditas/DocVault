"""Small deterministic checks for prefix-summary normalization."""

import pytest

from app.services.ai_types import TextChunk
from app.services.summary_service import DocumentSummaryService
from tests.fakes import FakeChatModel


def chunks() -> tuple[TextChunk, ...]:
    return (
        TextChunk(0, "paragraph", "Leave", "Employees receive annual leave."),
        TextChunk(1, "list", "Leave > Limits", "Five days may carry over."),
    )


async def test_summary_is_one_bounded_model_call() -> None:
    model = FakeChatModel(
        reply="First fact\n- Second fact\nThird fact\nFourth fact",
    )
    service = DocumentSummaryService(model, max_lines=3, max_chars=200)

    result = await service.summarize("Leave policy", chunks())

    assert model.calls == 1
    assert result == "- First fact\n- Second fact\n- Third fact"
    assert "beginning of the document" in model.last_user_prompt


async def test_summary_is_bounded_by_characters_as_well_as_lines() -> None:
    service = DocumentSummaryService(
        FakeChatModel(reply=f"- {'x' * 1000}"),
        max_lines=7,
        max_chars=100,
    )

    result = await service.summarize("Long line", chunks())

    assert len(result) == 100


async def test_empty_summary_response_is_rejected() -> None:
    service = DocumentSummaryService(FakeChatModel(reply="```\n```"))

    with pytest.raises(RuntimeError, match="empty summary"):
        await service.summarize("Empty", chunks())


async def test_no_chunks_are_rejected_without_calling_model() -> None:
    model = FakeChatModel()
    service = DocumentSummaryService(model)

    with pytest.raises(RuntimeError, match="without chunks"):
        await service.summarize("Empty", ())

    assert model.calls == 0
