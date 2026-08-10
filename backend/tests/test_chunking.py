"""Structure-aware chunking tests. No Docker and no model download."""

from app.config import Settings
from app.services.ai_types import ExtractedBlock, ExtractedDocument, SourceLocation
from app.services.chunking_service import ChunkingService
from app.services.text_extraction_service import _extract_sync


class WordCounter:
    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split()))


def _service(target: int = 12, maximum: int = 16, overlap: int = 4, minimum: int = 3):
    return ChunkingService(
        Settings(
            chunk_target_tokens=target,
            chunk_max_tokens=maximum,
            chunk_overlap_tokens=overlap,
            chunk_min_tokens=minimum,
        ),
        WordCounter(),
    )


def _block(text: str, page: int | None = None, start: int = 0) -> ExtractedBlock:
    return ExtractedBlock(
        text=text,
        block_type="paragraph",
        location=SourceLocation(page_number=page, char_start=start, char_end=start + len(text)),
    )


def test_short_document_is_one_chunk() -> None:
    chunks = _service().split_sync(ExtractedDocument((_block("a short policy"),)))
    assert [chunk.content for chunk in chunks] == ["a short policy"]


def test_chunks_never_split_an_ordinary_word() -> None:
    text = "alpha beta gamma notwithstanding regional requirements epsilon zeta eta theta"
    chunks = _service(target=5, maximum=6, overlap=1).split_sync(ExtractedDocument((_block(text),)))
    assert chunks
    assert all("notwith\n" not in chunk.content for chunk in chunks)
    assert any("notwithstanding" in chunk.content for chunk in chunks)


def test_chunks_do_not_apply_global_overlap() -> None:
    blocks = tuple(_block(f"Sentence number {i} has context.", start=i * 100) for i in range(6))
    chunks = _service(target=10, maximum=14, overlap=5).split_sync(ExtractedDocument(blocks))
    assert len(chunks) > 1
    rendered = "\n\n".join(chunk.content for chunk in chunks)
    for block in blocks:
        assert rendered.count(block.text) == 1


def test_pdf_chunk_can_cross_page_boundary_without_overlap() -> None:
    document = ExtractedDocument(
        (
            _block("Eligibility begins after", page=1, start=900),
            _block("six months of continuous employment.", page=2, start=0),
            _block("Requests need two weeks notice.", page=2, start=40),
        )
    )
    chunks = _service(target=10, maximum=14, overlap=5).split_sync(document)
    assert chunks[0].content == ("Eligibility begins after\n\nsix months of continuous employment.")
    assert {span.location.page_number for span in chunks[0].source_spans} == {1, 2}
    assert chunks[1].content == "Requests need two weeks notice."


def test_source_spans_point_into_chunk_content() -> None:
    document = ExtractedDocument((_block("first block", page=1), _block("second block", page=2)))
    chunk = _service(target=20).split_sync(document)[0]
    for span in chunk.source_spans:
        extracted = chunk.content[span.chunk_start : span.chunk_end]
        expected = document.blocks[span.sequence].text
        assert extracted == expected


def test_small_tail_is_merged_or_kept_never_dropped() -> None:
    document = ExtractedDocument(
        (_block("one two three four five six"), _block("HR approval required"))
    )
    chunks = _service(target=6, maximum=12, overlap=1, minimum=4).split_sync(document)
    assert "HR approval required" in "\n\n".join(chunk.content for chunk in chunks)


def test_heading_comes_from_latest_piece() -> None:
    blocks = (
        ExtractedBlock("# Leave", "heading", SourceLocation(heading="Leave")),
        ExtractedBlock("Annual leave applies.", "paragraph", SourceLocation(heading="Leave")),
    )
    assert _service(target=20).split_sync(ExtractedDocument(blocks))[0].heading == "Leave"


def test_chunking_is_deterministic() -> None:
    document = ExtractedDocument(tuple(_block(f"block {i} with words") for i in range(10)))
    service = _service()
    first = service.split_sync(document)
    second = service.split_sync(document)
    assert [
        (chunk.logical_key, chunk.chunk_type, chunk.content, chunk.embedding_text)
        for chunk in first
    ] == [
        (chunk.logical_key, chunk.chunk_type, chunk.content, chunk.embedding_text)
        for chunk in second
    ]


async def test_async_split_matches_sync() -> None:
    document = ExtractedDocument((_block("one two three"),))
    service = _service()
    asynchronous = await service.split(document)
    synchronous = service.split_sync(document)
    assert [(chunk.logical_key, chunk.content) for chunk in asynchronous] == [
        (chunk.logical_key, chunk.content) for chunk in synchronous
    ]


def test_chunks_never_cross_sibling_section_boundaries() -> None:
    artifact = _extract_sync(b"# First\n\nalpha details\n\n# Second\n\nbeta details", "sections.md")
    chunks = _service(target=30, maximum=40).split_sync(artifact)
    assert not any(
        "alpha details" in chunk.content and "beta details" in chunk.content for chunk in chunks
    )
    assert {chunk.breadcrumb for chunk in chunks} == {"First", "Second"}


def test_small_list_stays_with_its_introduction_and_markers() -> None:
    artifact = _extract_sync(b"# Leave\n\nBring:\n\n- ID\n- Form", "list.md")
    chunks = _service(target=30, maximum=40).split_sync(artifact)
    list_chunk = next(chunk for chunk in chunks if chunk.chunk_type == "list_chunk")
    assert list_chunk.content == "Bring:\n\n- ID\n\n- Form"


def test_contextualized_embedding_input_respects_the_hard_maximum() -> None:
    artifact = _extract_sync(
        ("# Long heading\n\n" + "sentence with several ordinary words. " * 30).encode(),
        "long.md",
    )
    service = _service(target=20, maximum=28)
    chunks = service.split_sync(artifact)
    assert chunks
    assert all(WordCounter().count_tokens(chunk.embedding_text) <= 28 for chunk in chunks)
