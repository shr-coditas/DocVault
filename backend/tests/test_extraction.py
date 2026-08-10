"""Logical-block extraction tests. No Docker or network."""

import io

import pytest
from docx import Document as DocxDocument

from app.config import Settings
from app.exceptions import UnsupportedMediaTypeError
from app.services.ai_types import SourceLocation
from app.services.text_extraction_service import (
    ExtractionError,
    TextExtractionService,
    _Builder,
    _extract_sync,
    _suppress_repeated_page_bands,
)


def _make_pdf(page_texts: list[str]) -> bytes:
    page_nums = [4 + 2 * i for i in range(len(page_texts))]
    content_nums = [5 + 2 * i for i in range(len(page_texts))]
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{' '.join(f'{n} 0 R' for n in page_nums)}] /Count {len(page_texts)} >>".encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for index, value in enumerate(page_texts):
        escaped = value.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        stream = f"BT /F1 12 Tf 20 150 Td ({escaped}) Tj ET".encode()
        objects[content_nums[index]] = (
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
        objects[page_nums[index]] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents {content_nums[index]} 0 R /Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
    output = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(output)
        output += f"{number} 0 obj\n".encode() + objects[number] + b"\nendobj\n"
    xref = len(output)
    highest = max(objects)
    output += f"xref\n0 {highest + 1}\n".encode() + b"0000000000 65535 f \n"
    for number in range(1, highest + 1):
        output += f"{offsets[number]:010d} 00000 n \n".encode()
    output += f"trailer\n<< /Size {highest + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(output)


def test_txt_groups_lines_into_a_normalised_paragraph() -> None:
    document = _extract_sync(b"line one\r\nline two", "notes.txt")
    assert [block.text for block in document.blocks] == ["line one line two"]
    assert document.blocks[0].location.line_start == 1
    assert document.blocks[0].location.line_end == 2


def test_markdown_carries_active_heading() -> None:
    document = _extract_sync(b"# Leave\n\nAnnual leave applies.", "readme.md")
    assert document.blocks[0].block_type == "heading"
    assert document.blocks[-1].location.heading == "Leave"


def test_csv_keeps_each_row_complete_and_self_describing() -> None:
    document = _extract_sync(b"name,dept\nAlice,Finance\nBob,HR\n", "people.csv")
    assert [block.text for block in document.blocks] == [
        "name: Alice; dept: Finance",
        "name: Bob; dept: HR",
    ]
    assert [block.location.row_start for block in document.blocks] == [2, 3]


def test_pdf_blocks_have_one_based_page_numbers() -> None:
    document = _extract_sync(_make_pdf(["alpha", "beta"]), "doc.pdf")
    assert [block.location.page_number for block in document.blocks] == [1, 2]


def test_docx_heading_and_paragraph_metadata() -> None:
    doc = DocxDocument()
    doc.add_paragraph("Leave Policy", style="Heading 1")
    doc.add_paragraph("Annual leave applies.")
    buffer = io.BytesIO()
    doc.save(buffer)
    document = _extract_sync(buffer.getvalue(), "policy.docx")
    assert [block.block_type for block in document.blocks] == ["heading", "paragraph"]
    assert document.blocks[1].location.heading == "Leave Policy"


@pytest.mark.parametrize("name", ["broken.pdf", "broken.docx"])
def test_corrupt_structured_file_raises(name: str) -> None:
    with pytest.raises(ExtractionError):
        _extract_sync(b"not a real document", name)


def test_unsupported_extension_is_rejected() -> None:
    with pytest.raises(UnsupportedMediaTypeError):
        _extract_sync(b"...", "archive.zip")


def test_supported_extensions_match_upload_allowlist() -> None:
    from app.services.text_extraction_service import _HANDLERS

    assert set(_HANDLERS) == set(Settings().allowed_upload_extensions)


async def test_service_returns_extracted_document() -> None:
    document = await TextExtractionService().extract(b"hello", "a.txt")
    assert document.blocks[0].text == "hello"
    assert not document.is_empty


def test_markdown_ast_preserves_nested_sections_lists_tables_and_code() -> None:
    document = _extract_sync(
        b"# Policy\n\nIntro:\n\n1. Parent\n   - Child\n\n| Key | Value |\n| --- | --- |\n| A | B |\n\n```py\ndef run(): pass\n```",
        "policy.md",
    )
    types = [node.node_type for node in document.nodes]
    assert {
        "document",
        "section",
        "heading",
        "list",
        "list_item",
        "table",
        "table_row",
        "table_cell",
        "code_block",
    } <= set(types)
    assert all(
        node.parent_id is not None for node in document.nodes if node.node_type not in {"document"}
    )


def test_page_band_suppression_requires_three_occurrences_and_sixty_percent() -> None:
    builder = _Builder(root_text="fixture.pdf")
    for page in range(1, 6):
        builder.add(
            "paragraph",
            builder.root_id,
            text="Confidential",
            source_spans=(SourceLocation(page_number=page),),
            attributes={"page_band": "header"},
        )
    builder.add(
        "paragraph",
        builder.root_id,
        text="One-off footer",
        source_spans=(SourceLocation(page_number=5),),
        attributes={"page_band": "footer"},
    )

    assert _suppress_repeated_page_bands(builder, 5) == 5
    assert sum(node.node_type == "suppressed_header" for node in builder.nodes) == 5
    assert any(
        node.text == "One-off footer" and node.node_type == "paragraph" for node in builder.nodes
    )
