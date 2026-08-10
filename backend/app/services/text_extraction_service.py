"""Extract supported files into a canonical hierarchical document artifact."""

import asyncio
import csv
import hashlib
import io
import os
import re
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from math import ceil
from pathlib import PurePosixPath
from typing import Protocol

import pypdf
from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph
from uuid6 import uuid7

from app.exceptions import UnsupportedMediaTypeError
from app.services.ai_types import (
    DocumentNode,
    ExtractedArtifact,
    ExtractedDocument,
    NodeType,
    SourceLocation,
)

_MARKDOWN_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
_LIST_ITEM = re.compile(r"^(\s*)(?:[-*+] |\d+[.)] |[a-zA-Z][.)] )")
_DOCX_HEADING = re.compile(r"^Heading\s+(\d+)$", re.IGNORECASE)
_SENTENCE_HYPHEN = re.compile(r"(?<=\w)-\n(?=[a-z])")
_SPACE = re.compile(r"[ \t]+")
_LOW_TEXT_CHARS = 20
_OCR_PAGE_RATIO = 0.8
_REPEATED_BAND_RATIO = 0.6
_REPEATED_BAND_MINIMUM = 3


class ExtractionError(Exception):
    """A supported file could not be parsed."""


class _Builder:
    def __init__(self, *, root_text: str) -> None:
        root_id: uuid.UUID = uuid7()
        self.root_id = root_id
        self.nodes: list[DocumentNode] = [
            DocumentNode(root_id, "document", "document", None, 0, text=root_text)
        ]
        self._ordinals: Counter[uuid.UUID] = Counter()

    def add(
        self,
        node_type: NodeType,
        parent_id: uuid.UUID,
        *,
        text: str | None = None,
        heading_level: int | None = None,
        source_spans: tuple[SourceLocation, ...] = (),
        confidence: float = 1.0,
        attributes: dict[str, object] | None = None,
    ) -> DocumentNode:
        ordinal = self._ordinals[parent_id]
        self._ordinals[parent_id] += 1
        parent = next(node for node in self.nodes if node.id == parent_id)
        path = f"{parent.logical_path}/{node_type}:{ordinal}"
        raw_text = text or ""
        normalized_text = _normalise_node_text(text) if text is not None else None
        node_attributes = dict(attributes or {})
        transformations = []
        if "\r" in raw_text:
            transformations.append("line_endings")
        if _SENTENCE_HYPHEN.search(_normalise(raw_text)):
            transformations.append("dehyphenation")
        if normalized_text is not None and normalized_text != _normalise(raw_text).strip():
            transformations.append("whitespace")
        if transformations:
            node_attributes["transformations"] = transformations
        normalized_spans = source_spans
        if normalized_text is not None and len(source_spans) == 1:
            normalized_spans = (
                replace(
                    source_spans[0],
                    normalized_char_start=0,
                    normalized_char_end=len(normalized_text),
                ),
            )
        node = DocumentNode(
            id=uuid7(),
            logical_path=path,
            node_type=node_type,
            parent_id=parent.id,
            ordinal=ordinal,
            text=normalized_text,
            heading_level=heading_level,
            source_spans=normalized_spans,
            confidence=confidence,
            attributes=node_attributes,
        )
        self.nodes.append(node)
        return node

    def update(
        self,
        node_id: uuid.UUID,
        *,
        text: str | None = None,
        attributes: dict[str, object] | None = None,
        node_type: NodeType | None = None,
    ) -> None:
        for index, node in enumerate(self.nodes):
            if node.id != node_id:
                continue
            merged = dict(node.attributes)
            if attributes:
                merged.update(attributes)
            new_text = node.text
            if text:
                new_text = f"{new_text}\n{text}" if new_text else text
            self.nodes[index] = replace(
                node,
                text=_normalise_node_text(new_text) if new_text else new_text,
                attributes=merged,
                node_type=node_type or node.node_type,
            )
            return
        raise KeyError(node_id)


def _suppress_repeated_page_bands(builder: _Builder, page_count: int) -> int:
    """Suppress only genuinely repeated top/bottom page-band content."""
    candidates: dict[tuple[str, str], list[uuid.UUID]] = {}
    for node in builder.nodes:
        band = node.attributes.get("page_band")
        if band not in {"header", "footer"} or not node.text:
            continue
        normalized = " ".join(node.text.casefold().split())
        candidates.setdefault((str(band), normalized), []).append(node.id)

    required = max(_REPEATED_BAND_MINIMUM, ceil(page_count * _REPEATED_BAND_RATIO))
    suppressed: set[uuid.UUID] = set()
    for ids in candidates.values():
        pages = {
            span.page_number
            for node in builder.nodes
            if node.id in ids
            for span in node.source_spans
            if span.page_number is not None
        }
        if len(ids) >= _REPEATED_BAND_MINIMUM and len(pages) >= required:
            suppressed.update(ids)

    for index, node in enumerate(builder.nodes):
        if node.id not in suppressed:
            continue
        band = str(node.attributes["page_band"])
        builder.nodes[index] = replace(
            node,
            node_type="suppressed_header" if band == "header" else "suppressed_footer",
            attributes={**node.attributes, "suppression_rule": "repeated_page_band_v1"},
        )
    return len(suppressed)


def _package_version(package: str, fallback: str = "unknown") -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return fallback


def _normalise(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _normalise_node_text(text: str | None) -> str:
    if not text:
        return ""
    value = _normalise(text)
    value = _SENTENCE_HYPHEN.sub("", value)
    return "\n".join(_SPACE.sub(" ", line).strip() for line in value.splitlines()).strip()


def _decode(data: bytes) -> str:
    for encoding in ("utf-8", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _deterministic_scalar_type(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return "empty"
    if stripped.casefold() in {"true", "false"}:
        return "boolean"
    if re.fullmatch(r"[+-]?\d+", stripped):
        return "integer"
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)", stripped):
        return "decimal"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
        return "date"
    return "string"


def _location_from_docling(item: object) -> tuple[SourceLocation, ...]:
    locations: list[SourceLocation] = []
    for provenance in getattr(item, "prov", ()) or ():
        bbox_object = getattr(provenance, "bbox", None)
        bbox = None
        if bbox_object is not None:
            left, top, right, bottom = (
                getattr(bbox_object, name, None) for name in ("l", "t", "r", "b")
            )
            if (
                isinstance(left, int | float)
                and isinstance(top, int | float)
                and isinstance(right, int | float)
                and isinstance(bottom, int | float)
            ):
                bbox = (float(left), float(top), float(right), float(bottom))
        page = getattr(provenance, "page_no", None)
        locations.append(SourceLocation(page_number=page, bbox=bbox))
    return tuple(locations)


def _artifact(
    builder: _Builder,
    *,
    parser_name: str,
    parser_version: str,
    data: bytes,
    file_name: str,
    warnings: tuple[str, ...] = (),
    quality_metrics: dict[str, object] | None = None,
    raw_payload: dict[str, object] | None = None,
) -> ExtractedArtifact:
    return ExtractedArtifact(
        nodes=tuple(builder.nodes),
        parser_name=parser_name,
        parser_version=parser_version,
        source_checksum=hashlib.sha256(data).hexdigest(),
        metadata={"file_name": file_name},
        warnings=warnings,
        quality_metrics=quality_metrics or {},
        raw_payload=raw_payload,
    )


def artifact_from_blocks(
    extracted: ExtractedDocument, *, document_title: str = "Document"
) -> ExtractedArtifact:
    """Lift the old flat contract into the AST for compatibility and tests."""
    builder = _Builder(root_text=document_title)
    current_section: uuid.UUID = builder.root_id
    current_heading: str | None = None
    for block in extracted.blocks:
        if block.block_type == "heading":
            current_heading = block.location.heading or block.text.lstrip("# ")
            section = builder.add("section", builder.root_id, text=current_heading, heading_level=1)
            builder.add(
                "heading",
                section.id,
                text=block.text,
                heading_level=1,
                source_spans=(block.location,),
            )
            current_section = section.id
            continue
        node_type: NodeType = "list_item" if block.block_type == "list_item" else "paragraph"
        builder.add(
            node_type,
            current_section,
            text=block.text,
            source_spans=(replace(block.location, heading=current_heading),),
        )
    return ExtractedArtifact(
        tuple(builder.nodes),
        "legacy-flat",
        "1",
        "",
        metadata={"document_title": document_title},
    )


def _extract_pdf_fallback(data: bytes, file_name: str, reason: str) -> ExtractedArtifact:
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        builder = _Builder(root_text=file_name)
        page_char_counts: list[int] = []
        for page_number, page in enumerate(reader.pages, start=1):
            if page_number > 1:
                builder.add(
                    "page_break",
                    builder.root_id,
                    attributes={"page_number": page_number},
                )
            text = _normalise(page.extract_text() or "")
            page_char_counts.append(sum(character.isalnum() for character in text))
            offset = 0
            nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
            first_line = nonempty_lines[0] if nonempty_lines else None
            last_line = nonempty_lines[-1] if nonempty_lines else None
            for line_number, raw_line in enumerate(text.splitlines(keepends=True), start=1):
                line = raw_line.rstrip("\n")
                stripped = line.strip()
                start = offset
                end = start + len(line)
                offset += len(raw_line)
                if not stripped:
                    continue
                location = SourceLocation(
                    page_number=page_number,
                    line_start=line_number,
                    line_end=line_number,
                    char_start=start,
                    char_end=end,
                )
                builder.add(
                    "paragraph",
                    builder.root_id,
                    text=line,
                    source_spans=(location,),
                    confidence=0.5,
                    attributes={
                        "fallback_text": True,
                        **(
                            {"page_band": "header"}
                            if stripped == first_line and len(nonempty_lines) > 1
                            else {"page_band": "footer"}
                            if stripped == last_line and len(nonempty_lines) > 1
                            else {}
                        ),
                    },
                )
        low_pages = sum(count < _LOW_TEXT_CHARS for count in page_char_counts)
        ratio = low_pages / len(page_char_counts) if page_char_counts else 1.0
        warnings = ["pdf_layout_fallback"]
        if ratio >= _OCR_PAGE_RATIO:
            warnings.append("ocr_required")
        suppressed = _suppress_repeated_page_bands(builder, len(page_char_counts))
        return _artifact(
            builder,
            parser_name="pypdf-fallback",
            parser_version=_package_version("pypdf"),
            data=data,
            file_name=file_name,
            warnings=tuple(warnings),
            quality_metrics={
                "page_count": len(page_char_counts),
                "low_text_pages": low_pages,
                "low_text_page_ratio": ratio,
                "fallback_reason": reason,
                "suppressed_page_band_nodes": suppressed,
            },
        )
    except Exception as exc:
        raise ExtractionError(f"could not read PDF: {exc}") from exc


def _docling_table_text(item: object, document: object) -> tuple[str, list[str]]:
    try:
        frame = item.export_to_dataframe(doc=document)  # type: ignore[attr-defined]
        headers = [str(column) for column in frame.columns]
        lines = [" | ".join(headers)]
        lines.extend(
            " | ".join(str(value) for value in row) for row in frame.itertuples(index=False)
        )
        return "\n".join(lines), headers
    except Exception:
        text = str(getattr(item, "text", "") or "")
        return text, []


def _extract_pdf_docling(data: bytes, file_name: str) -> ExtractedArtifact:
    from docling.datamodel.base_models import DocumentStream, InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = False
    options.do_table_structure = True
    options.enable_remote_services = False
    options.artifacts_path = os.environ.get("DOCLING_ARTIFACTS_PATH")
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    result = converter.convert(DocumentStream(name=file_name, stream=io.BytesIO(data)))
    document = result.document
    builder = _Builder(root_text=file_name)
    sections: list[tuple[int, uuid.UUID]] = [(0, builder.root_id)]
    page_numbers: set[int] = set()
    page_text: Counter[int] = Counter()

    for item, level in document.iterate_items():
        raw_label = getattr(item, "label", "text")
        label = str(getattr(raw_label, "value", raw_label)).lower()
        text = str(getattr(item, "text", "") or "").strip()
        locations = _location_from_docling(item)
        for location in locations:
            if location.page_number is not None:
                page_numbers.add(location.page_number)
                page_text[location.page_number] += sum(character.isalnum() for character in text)

        if label in {"title", "section_header"}:
            heading_level = max(1, int(level or 1))
            while sections and sections[-1][0] >= heading_level:
                sections.pop()
            parent_id = sections[-1][1] if sections else builder.root_id
            section = builder.add(
                "section", parent_id, text=text, heading_level=heading_level, source_spans=locations
            )
            builder.add(
                "heading",
                section.id,
                text=text,
                heading_level=heading_level,
                source_spans=locations,
            )
            sections.append((heading_level, section.id))
            continue

        parent_id = sections[-1][1] if sections else builder.root_id
        if label == "table":
            table_text, headers = _docling_table_text(item, document)
            builder.add(
                "table",
                parent_id,
                text=table_text,
                source_spans=locations,
                attributes={"headers": headers, "layout_aware": True},
            )
        elif label in {"list_item"}:
            builder.add("list_item", parent_id, text=text, source_spans=locations)
        elif label in {"code"}:
            builder.add("code_block", parent_id, text=text, source_spans=locations)
        elif label in {"caption"}:
            builder.add("caption", parent_id, text=text, source_spans=locations)
        elif label in {"picture"}:
            builder.add("figure", parent_id, text=text or None, source_spans=locations)
        elif label in {"page_header", "page_footer"}:
            builder.add(
                "paragraph",
                parent_id,
                text=text,
                source_spans=locations,
                attributes={"page_band": "header" if label == "page_header" else "footer"},
            )
        elif text:
            builder.add("paragraph", parent_id, text=text, source_spans=locations)

    page_count = len(page_numbers)
    low_pages = sum(page_text[page] < _LOW_TEXT_CHARS for page in page_numbers)
    ratio = low_pages / page_count if page_count else 1.0
    suppressed = _suppress_repeated_page_bands(builder, page_count)
    warnings = ("ocr_required",) if ratio >= _OCR_PAGE_RATIO else ()
    return _artifact(
        builder,
        parser_name="docling",
        parser_version=_package_version("docling"),
        data=data,
        file_name=file_name,
        warnings=warnings,
        quality_metrics={
            "page_count": page_count,
            "low_text_pages": low_pages,
            "low_text_page_ratio": ratio,
            "layout_aware": True,
            "suppressed_page_band_nodes": suppressed,
        },
        raw_payload=document.export_to_dict(),
    )


def _extract_pdf(data: bytes, file_name: str) -> ExtractedArtifact:
    try:
        return _extract_pdf_docling(data, file_name)
    except ModuleNotFoundError:
        return _extract_pdf_fallback(data, file_name, "docling_not_installed")
    except Exception as exc:
        return _extract_pdf_fallback(data, file_name, f"{type(exc).__name__}: {exc}")


def _docx_list_level(paragraph: Paragraph) -> int:
    num_properties = getattr(getattr(paragraph._p, "pPr", None), "numPr", None)
    level = getattr(getattr(num_properties, "ilvl", None), "val", None)
    return int(level) if level is not None else 0


def _docx_list_marker(style: str) -> str:
    return "- " if "Bullet" in style else "1. "


def _extract_docx(data: bytes, file_name: str) -> ExtractedArtifact:
    try:
        document = DocxDocument(io.BytesIO(data))
        builder = _Builder(root_text=file_name)
        sections: list[tuple[int, uuid.UUID]] = [(0, builder.root_id)]
        current_list: uuid.UUID | None = None
        paragraph_index = 0
        pending_caption: str | None = None

        contents = (
            document.iter_inner_content()
            if hasattr(document, "iter_inner_content")
            else document.paragraphs
        )
        for item in contents:
            parent_id = sections[-1][1]
            if isinstance(item, Paragraph):
                text = _normalise_node_text(item.text)
                style = item.style.name if item.style is not None else ""
                location = SourceLocation(paragraph_index=paragraph_index)
                paragraph_index += 1
                if not text and not item._p.xpath(".//w:drawing"):
                    current_list = None
                    continue
                heading_match = _DOCX_HEADING.match(style)
                if heading_match or style in {"Title", "Subtitle"}:
                    level = int(heading_match.group(1)) if heading_match else 1
                    while sections and sections[-1][0] >= level:
                        sections.pop()
                    parent_id = sections[-1][1] if sections else builder.root_id
                    section = builder.add("section", parent_id, text=text, heading_level=level)
                    builder.add(
                        "heading",
                        section.id,
                        text=text,
                        heading_level=level,
                        source_spans=(location,),
                    )
                    sections.append((level, section.id))
                    current_list = None
                elif style.startswith("Caption"):
                    pending_caption = text
                    builder.add("caption", parent_id, text=text, source_spans=(location,))
                    current_list = None
                elif style.startswith(("List", "Bullet")) or _LIST_ITEM.match(text):
                    if current_list is None:
                        current_list = builder.add("list", parent_id).id
                    builder.add(
                        "list_item",
                        current_list,
                        text=text,
                        source_spans=(location,),
                        attributes={
                            "level": _docx_list_level(item),
                            "style": style,
                            "marker": _docx_list_marker(style),
                        },
                    )
                else:
                    current_list = None
                    if item._p.xpath(".//w:drawing"):
                        builder.add(
                            "figure",
                            parent_id,
                            text=None,
                            source_spans=(location,),
                            attributes={"inline_shape": True},
                        )
                    if text:
                        builder.add("paragraph", parent_id, text=text, source_spans=(location,))
                continue

            if isinstance(item, Table):
                attributes: dict[str, object] = {"column_count": len(item.columns)}
                if pending_caption:
                    attributes["caption"] = pending_caption
                    pending_caption = None
                table = builder.add("table", parent_id, attributes=attributes)
                headers: list[str] = []
                rendered_rows: list[str] = []
                for row_index, row in enumerate(item.rows, start=1):
                    values = [_normalise_node_text(cell.text) for cell in row.cells]
                    if row_index == 1:
                        headers = values
                    row_node = builder.add(
                        "table_row",
                        table.id,
                        text=" | ".join(values),
                        source_spans=(SourceLocation(row_start=row_index, row_end=row_index),),
                        attributes={"row_index": row_index, "header": row_index == 1},
                    )
                    for column_index, value in enumerate(values):
                        builder.add(
                            "table_cell",
                            row_node.id,
                            text=value,
                            attributes={"column_index": column_index},
                        )
                    rendered_rows.append(" | ".join(values))
                builder.update(
                    table.id, text="\n".join(rendered_rows), attributes={"headers": headers}
                )
                current_list = None

        return _artifact(
            builder,
            parser_name="python-docx",
            parser_version=_package_version("python-docx"),
            data=data,
            file_name=file_name,
            quality_metrics={"paragraph_count": paragraph_index},
        )
    except Exception as exc:
        raise ExtractionError(f"could not read DOCX: {exc}") from exc


def _extract_csv(data: bytes, file_name: str) -> ExtractedArtifact:
    text = _normalise(_decode(data))
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error as exc:
        raise ExtractionError(f"could not read CSV: {exc}") from exc
    builder = _Builder(root_text=file_name)
    if not rows:
        return _artifact(
            builder, parser_name="csv", parser_version="stdlib", data=data, file_name=file_name
        )
    headers, *body = rows
    table = builder.add(
        "table", builder.root_id, attributes={"headers": headers, "column_count": len(headers)}
    )
    source_rows = body or [headers]
    rendered_rows: list[str] = []
    for row_number, row in enumerate(source_rows, start=2 if body else 1):
        values = list(row)
        rendered = "; ".join(
            f"{name}: {value}" for name, value in zip(headers, values, strict=False) if value
        )
        row_node = builder.add(
            "table_row",
            table.id,
            text=rendered or ", ".join(values),
            source_spans=(SourceLocation(row_start=row_number, row_end=row_number),),
            attributes={"row_index": row_number, "header": not body},
        )
        for column_index, value in enumerate(values):
            builder.add(
                "table_cell",
                row_node.id,
                text=value,
                attributes={
                    "column_index": column_index,
                    "header": headers[column_index] if column_index < len(headers) else None,
                    "inferred_type": _deterministic_scalar_type(value),
                },
            )
        if rendered:
            rendered_rows.append(rendered)
    builder.update(table.id, text="\n".join(rendered_rows))
    return _artifact(
        builder,
        parser_name="csv",
        parser_version="stdlib",
        data=data,
        file_name=file_name,
        quality_metrics={"row_count": len(source_rows), "column_count": len(headers)},
    )


def _extract_markdown(data: bytes, file_name: str) -> ExtractedArtifact:
    text = _normalise(_decode(data))
    try:
        from markdown_it import MarkdownIt
    except ModuleNotFoundError:
        return _extract_plain(data, file_name, markdown_fallback=True)

    parser = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = parser.parse(text)
    builder = _Builder(root_text=file_name)
    sections: list[tuple[int, uuid.UUID]] = [(0, builder.root_id)]
    containers: list[tuple[str, uuid.UUID]] = []
    current_cell: uuid.UUID | None = None
    current_row: uuid.UUID | None = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        token_type = token.type
        parent_id = containers[-1][1] if containers else sections[-1][1]
        line = token.map[0] if token.map is not None else None
        location = SourceLocation(line_start=line + 1 if line is not None else None)
        if token_type == "heading_open":
            level = int(token.tag[1:])
            inline = tokens[index + 1]
            heading = inline.content.strip()
            while sections and sections[-1][0] >= level:
                sections.pop()
            section_parent = sections[-1][1] if sections else builder.root_id
            section = builder.add("section", section_parent, text=heading, heading_level=level)
            builder.add(
                "heading", section.id, text=heading, heading_level=level, source_spans=(location,)
            )
            sections.append((level, section.id))
            index += 3
            continue
        if token_type in {"bullet_list_open", "ordered_list_open"}:
            marker = "ordered" if token_type.startswith("ordered") else "bullet"
            node = builder.add(
                "list",
                parent_id,
                attributes={"marker": marker, "start": int(token.attrGet("start") or 1)},
            )
            containers.append(("list", node.id))
        elif token_type in {"bullet_list_close", "ordered_list_close"}:
            if containers and containers[-1][0] == "list":
                containers.pop()
        elif token_type == "list_item_open":
            parent = next(node for node in builder.nodes if node.id == parent_id)
            raw_start = parent.attributes.get("start", 1)
            list_start = raw_start if isinstance(raw_start, int) else 1
            item_count = sum(
                1
                for child in builder.nodes
                if child.parent_id == parent_id and child.node_type == "list_item"
            )
            marker = (
                f"{list_start + item_count}. "
                if parent.attributes.get("marker") == "ordered"
                else "- "
            )
            node = builder.add(
                "list_item", parent_id, source_spans=(location,), attributes={"marker": marker}
            )
            containers.append(("list_item", node.id))
        elif token_type == "list_item_close":
            if containers and containers[-1][0] == "list_item":
                containers.pop()
        elif token_type == "blockquote_open":
            node = builder.add("quote", parent_id, source_spans=(location,))
            containers.append(("quote", node.id))
        elif token_type == "blockquote_close":
            if containers and containers[-1][0] == "quote":
                containers.pop()
        elif token_type == "table_open":
            node = builder.add("table", parent_id)
            containers.append(("table", node.id))
        elif token_type == "table_close":
            if containers and containers[-1][0] == "table":
                containers.pop()
        elif token_type == "tr_open":
            row = builder.add("table_row", parent_id, source_spans=(location,))
            containers.append(("table_row", row.id))
            current_row = row.id
        elif token_type == "tr_close":
            if containers and containers[-1][0] == "table_row":
                containers.pop()
            current_row = None
        elif token_type in {"th_open", "td_open"}:
            cell = builder.add(
                "table_cell", parent_id, attributes={"header": token_type == "th_open"}
            )
            containers.append(("table_cell", cell.id))
            current_cell = cell.id
        elif token_type in {"th_close", "td_close"}:
            if containers and containers[-1][0] == "table_cell":
                containers.pop()
            current_cell = None
        elif token_type == "fence":
            builder.add(
                "code_block",
                parent_id,
                text=token.content,
                source_spans=(location,),
                attributes={"language": token.info.strip() or None},
            )
        elif token_type == "inline" and token.content.strip():
            if current_cell is not None:
                builder.update(current_cell, text=token.content)
                if current_row is not None:
                    builder.update(current_row, text=token.content)
            elif containers and containers[-1][0] in {"list_item", "quote"}:
                builder.update(containers[-1][1], text=token.content)
            elif index == 0 or tokens[index - 1].type != "heading_open":
                builder.add("paragraph", parent_id, text=token.content, source_spans=(location,))
        index += 1

    return _artifact(
        builder,
        parser_name="markdown-it-py",
        parser_version=_package_version("markdown-it-py"),
        data=data,
        file_name=file_name,
        raw_payload={"tokens": [token.as_dict() for token in tokens]},
    )


def _extract_plain(
    data: bytes, file_name: str, *, markdown_fallback: bool = False
) -> ExtractedArtifact:
    text = _normalise(_decode(data))
    builder = _Builder(root_text=file_name)
    parent_id: uuid.UUID = builder.root_id
    current_list: uuid.UUID | None = None
    offset = 0
    paragraph_lines: list[tuple[str, int, int, int]] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        rendered = " ".join(line.strip() for line, _, _, _ in paragraph_lines)
        first = paragraph_lines[0]
        last = paragraph_lines[-1]
        builder.add(
            "paragraph",
            parent_id,
            text=rendered,
            source_spans=(
                SourceLocation(
                    line_start=first[3],
                    line_end=last[3],
                    char_start=first[1],
                    char_end=last[2],
                ),
            ),
            confidence=0.7,
            attributes={"markdown_fallback": markdown_fallback},
        )
        paragraph_lines = []

    for line_number, raw_line in enumerate(text.splitlines(keepends=True), start=1):
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        start = offset
        end = start + len(line)
        offset += len(raw_line)
        if not stripped:
            flush_paragraph()
            current_list = None
            continue
        heading_match = _MARKDOWN_HEADING.match(line) if markdown_fallback else None
        probable_heading = (
            len(stripped) <= 100
            and any(character.isalpha() for character in stripped)
            and stripped == stripped.upper()
        )
        if heading_match or probable_heading:
            flush_paragraph()
            heading = heading_match.group(2) if heading_match else stripped
            level = len(heading_match.group(1)) if heading_match else 1
            section = builder.add("section", builder.root_id, text=heading, heading_level=level)
            builder.add(
                "heading",
                section.id,
                text=heading,
                heading_level=level,
                source_spans=(
                    SourceLocation(line_start=line_number, char_start=start, char_end=end),
                ),
                confidence=0.6,
            )
            parent_id = section.id
            current_list = None
        elif _LIST_ITEM.match(line):
            flush_paragraph()
            if current_list is None:
                current_list = builder.add("list", parent_id, confidence=0.6).id
            builder.add(
                "list_item",
                current_list,
                text=line,
                source_spans=(
                    SourceLocation(line_start=line_number, char_start=start, char_end=end),
                ),
                confidence=0.6,
            )
        else:
            paragraph_lines.append((line, start, end, line_number))
    flush_paragraph()
    parser_name = "markdown-fallback" if markdown_fallback else "plain-text"
    warnings = ("markdown_parser_unavailable",) if markdown_fallback else ()
    return _artifact(
        builder,
        parser_name=parser_name,
        parser_version="1",
        data=data,
        file_name=file_name,
        warnings=warnings,
    )


Handler = Callable[[bytes, str], ExtractedArtifact]
_HANDLERS: dict[str, Handler] = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".csv": _extract_csv,
    ".txt": _extract_plain,
    ".md": _extract_markdown,
}


def _extract_sync(data: bytes, file_name: str) -> ExtractedArtifact:
    suffix = PurePosixPath(file_name).suffix.lower()
    handler = _HANDLERS.get(suffix)
    if handler is None:
        raise UnsupportedMediaTypeError(
            f"cannot extract text from '{suffix or file_name}'; "
            f"supported: {', '.join(sorted(_HANDLERS))}"
        )
    return handler(data, file_name)


class DocumentExtractor(Protocol):
    async def extract(self, data: bytes, file_name: str) -> ExtractedArtifact: ...


class TextExtractionService:
    profile_name = "structured-extract-v2"

    async def extract(self, data: bytes, file_name: str) -> ExtractedArtifact:
        return await asyncio.to_thread(_extract_sync, data, file_name)
