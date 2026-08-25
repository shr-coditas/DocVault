"""Markdown-first chunking tests. No Docker and no model download."""

from app.config import Settings
from app.services.ai_types import ExtractedMarkdown, MarkdownBlock
from app.services.chunking_service import (
    ChunkingService,
    build_embedding_text,
    normalise_for_model,
)
from app.services.text_extraction_service import _extract_sync


class WordCounter:
    model_name = "word-counter"
    dimensions = 384

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split()))


def _service(maximum: int = 16) -> ChunkingService:
    return ChunkingService(Settings(chunk_max_tokens=maximum), WordCounter())  # type: ignore[arg-type]


def _document(*blocks: MarkdownBlock) -> ExtractedMarkdown:
    return ExtractedMarkdown(blocks)


def test_short_document_is_one_chunk() -> None:
    chunks = _service().split_sync(_document(MarkdownBlock("paragraph", "a short policy")))
    assert [chunk.content for chunk in chunks] == ["a short policy"]


def test_chunks_never_split_an_ordinary_word() -> None:
    text = "alpha beta gamma notwithstanding regional requirements epsilon zeta eta theta"
    chunks = _service(7).split_sync(_document(MarkdownBlock("paragraph", text)))
    assert chunks
    assert any("notwithstanding" in chunk.content for chunk in chunks)
    assert all("notwith standing" not in chunk.content for chunk in chunks)


def test_forced_word_windows_overlap_to_keep_context() -> None:
    text = " ".join(f"word{i}" for i in range(30))
    chunks = _service(9).split_sync(_document(MarkdownBlock("paragraph", text)))
    assert len(chunks) > 1
    previous = chunks[0].content.split()
    following = chunks[1].content.split()
    assert set(previous) & set(following)


def test_chunks_do_not_cross_section_boundaries() -> None:
    artifact = _extract_sync(
        b"# First\n\nalpha details\n\n# Second\n\nbeta details",
        "sections.md",
    )
    chunks = _service(30).split_sync(artifact)
    assert not any(
        "alpha details" in chunk.content and "beta details" in chunk.content for chunk in chunks
    )
    assert {chunk.section_path for chunk in chunks} == {"First", "Second"}


def test_small_list_keeps_complete_items_and_markers() -> None:
    artifact = _extract_sync(b"# Leave\n\n- ID\n- Form", "list.md")
    chunks = _service(30).split_sync(artifact)
    list_chunk = next(chunk for chunk in chunks if chunk.chunk_type == "list")
    assert list_chunk.content == "- ID\n- Form"


def test_large_table_splits_between_rows_and_repeats_header() -> None:
    artifact = _extract_sync(
        b"# Limits\n\n| Name | Value |\n| --- | --- |\n"
        b"| Alpha | one two three |\n| Beta | four five six |\n"
        b"| Gamma | seven eight nine |",
        "table.md",
    )
    chunks = _service(12).split_sync(artifact)
    tables = [chunk for chunk in chunks if chunk.chunk_type == "table"]
    assert len(tables) > 1
    assert all(chunk.content.startswith("Name | Value") for chunk in tables)


def test_embedding_input_respects_the_exact_hard_maximum() -> None:
    artifact = _extract_sync(
        ("# Long heading\n\n" + "sentence with several ordinary words. " * 30).encode(),
        "long.md",
    )
    service = _service(18)
    chunks = service.split_sync(artifact, document_title="Policy")
    assert chunks
    assert all(
        WordCounter().count_tokens(build_embedding_text("Policy", chunk)) <= 18 for chunk in chunks
    )


def test_long_url_is_preserved_but_normalised_for_embedding() -> None:
    url = "https://example.com/" + "very-long-path/" * 100
    chunks = _service(20).split_sync(_document(MarkdownBlock("paragraph", f"Policy link: {url}")))
    assert url in chunks[0].content
    normalised = normalise_for_model(chunks[0].content)
    assert "External link to example.com" in normalised
    assert url not in normalised


def test_abnormal_single_token_is_not_split_into_fragments() -> None:
    token = "x" * 1000
    chunks = _service(10).split_sync(_document(MarkdownBlock("paragraph", token)))
    assert [chunk.content for chunk in chunks] == [token]


def test_chunking_is_deterministic_and_async_matches_sync() -> None:
    document = _document(
        *(MarkdownBlock("paragraph", f"block {index} with words") for index in range(10))
    )
    service = _service()
    first = service.split_sync(document)
    second = service.split_sync(document)
    assert first == second


async def test_async_split_matches_sync() -> None:
    document = _document(MarkdownBlock("paragraph", "one two three"))
    service = _service()
    assert await service.split(document) == service.split_sync(document)
