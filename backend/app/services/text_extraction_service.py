"""Convert supported files into simple Markdown-style content blocks."""

import asyncio
import csv
import io
import re
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Any

import pypdf

from app.exceptions import UnsupportedMediaTypeError
from app.services.ai_types import ExtractedMarkdown, MarkdownBlock

_HEADING_STYLE = re.compile(r"^Heading\s+([1-6])$", re.IGNORECASE)
_LIST_STYLE = re.compile(r"^(List|Bullet|Number)", re.IGNORECASE)


class ExtractionError(Exception):
    """The file type is supported but its content could not be read."""


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _normalise(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _section_path(headings: list[tuple[int, str]]) -> str | None:
    return " > ".join(text for _, text in headings) or None


def _plain_blocks(text: str, section_path: str | None = None) -> list[MarkdownBlock]:
    blocks: list[MarkdownBlock] = []
    lines: list[str] = []

    def flush() -> None:
        if not lines:
            return
        content = " ".join(line.strip() for line in lines if line.strip())
        if content:
            blocks.append(MarkdownBlock("paragraph", content, section_path))
        lines.clear()

    for line in _normalise(text).splitlines():
        if line.strip():
            lines.append(line)
        else:
            flush()
    flush()
    return blocks


def _extract_plain(data: bytes, file_name: str) -> ExtractedMarkdown:
    del file_name
    return ExtractedMarkdown(tuple(_plain_blocks(_decode(data))))


def _extract_pdf(data: bytes, file_name: str) -> ExtractedMarkdown:
    del file_name
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        blocks: list[MarkdownBlock] = []
        for page in reader.pages:
            blocks.extend(_plain_blocks(page.extract_text() or ""))
        return ExtractedMarkdown(tuple(blocks))
    except Exception as exc:
        raise ExtractionError(f"could not read PDF: {exc}") from exc


def _extract_csv(data: bytes, file_name: str) -> ExtractedMarkdown:
    del file_name
    try:
        rows = list(csv.reader(io.StringIO(_normalise(_decode(data)))))
    except csv.Error as exc:
        raise ExtractionError(f"could not read CSV: {exc}") from exc
    content = "\n".join(" | ".join(cell.strip() for cell in row) for row in rows if row)
    blocks = (MarkdownBlock("table", content),) if content else ()
    return ExtractedMarkdown(blocks)


def _extract_docx(data: bytes, file_name: str) -> ExtractedMarkdown:
    del file_name
    try:
        from docx import Document as DocxDocument
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        document = DocxDocument(io.BytesIO(data))
        headings: list[tuple[int, str]] = []
        blocks: list[MarkdownBlock] = []
        list_items: list[str] = []
        list_path: str | None = None

        def flush_list() -> None:
            nonlocal list_path
            if list_items:
                blocks.append(MarkdownBlock("list", "\n".join(list_items), list_path))
                list_items.clear()
            list_path = None

        contents = (
            document.iter_inner_content()
            if hasattr(document, "iter_inner_content")
            else document.paragraphs
        )
        for item in contents:
            if isinstance(item, Paragraph):
                text = _normalise(item.text)
                style = item.style.name if item.style is not None else ""
                if not text:
                    flush_list()
                    continue
                heading = _HEADING_STYLE.match(style)
                if heading or style in {"Title", "Subtitle"}:
                    flush_list()
                    level = int(heading.group(1)) if heading else 1
                    headings[:] = [
                        (old_level, value) for old_level, value in headings if old_level < level
                    ]
                    headings.append((level, text))
                elif _LIST_STYLE.match(style):
                    if not list_items:
                        list_path = _section_path(headings)
                    marker = "1." if "number" in style.lower() else "-"
                    list_items.append(f"{marker} {text}")
                else:
                    flush_list()
                    blocks.append(MarkdownBlock("paragraph", text, _section_path(headings)))
                continue

            if isinstance(item, Table):
                flush_list()
                rows = [
                    " | ".join(_normalise(cell.text) for cell in row.cells) for row in item.rows
                ]
                content = "\n".join(row for row in rows if row.strip(" |"))
                if content:
                    blocks.append(MarkdownBlock("table", content, _section_path(headings)))
        flush_list()
        return ExtractedMarkdown(tuple(blocks))
    except Exception as exc:
        raise ExtractionError(f"could not read DOCX: {exc}") from exc


def _render_inline(token: Any) -> str:
    children = token.children or []
    if not children:
        return token.content.strip()
    rendered: list[str] = []
    links: list[str] = []
    for child in children:
        if child.type == "link_open":
            rendered.append("[")
            links.append(str(child.attrGet("href") or ""))
        elif child.type == "link_close":
            href = links.pop() if links else ""
            rendered.append(f"]({href})")
        elif child.type == "image":
            source = str(child.attrGet("src") or "")
            rendered.append(f"![{child.content}]({source})")
        elif child.type in {"softbreak", "hardbreak"}:
            rendered.append("\n")
        elif child.type in {"text", "code_inline", "html_inline"}:
            rendered.append(child.content)
    return "".join(rendered).strip()


def _extract_markdown(data: bytes, file_name: str) -> ExtractedMarkdown:
    del file_name
    try:
        from markdown_it import MarkdownIt
    except ModuleNotFoundError as exc:
        raise ExtractionError("markdown-it-py is required to read Markdown") from exc

    tokens = MarkdownIt("commonmark", {"html": False}).enable("table").parse(_decode(data))
    headings: list[tuple[int, str]] = []
    blocks: list[MarkdownBlock] = []
    list_stack: list[dict[str, int | bool]] = []
    item_stack: list[dict[str, Any]] = []
    list_lines: list[str] = []
    list_path: str | None = None
    table_rows: list[str] = []
    row_cells: list[str] | None = None
    index = 0

    while index < len(tokens):
        token = tokens[index]
        token_type = token.type
        if token_type == "heading_open" and index + 1 < len(tokens):
            level = int(token.tag[1:])
            heading = _render_inline(tokens[index + 1])
            headings[:] = [(old_level, text) for old_level, text in headings if old_level < level]
            headings.append((level, heading))
            index += 3
            continue

        if token_type in {"bullet_list_open", "ordered_list_open"}:
            if not list_stack:
                list_path = _section_path(headings)
            ordered = token_type == "ordered_list_open"
            start = int(token.attrGet("start") or 1) if ordered else 1
            list_stack.append({"ordered": ordered, "next": start})
        elif token_type in {"bullet_list_close", "ordered_list_close"}:
            if list_stack:
                list_stack.pop()
            if not list_stack and list_lines:
                blocks.append(MarkdownBlock("list", "\n".join(list_lines), list_path))
                list_lines.clear()
                list_path = None
        elif token_type == "list_item_open" and list_stack:
            current = list_stack[-1]
            number = int(current["next"])
            marker = f"{number}." if current["ordered"] else "-"
            current["next"] = number + 1
            item_stack.append(
                {"marker": marker, "depth": len(list_stack) - 1, "text": [], "children": []}
            )
        elif token_type == "list_item_close" and item_stack:
            item = item_stack.pop()
            text = " ".join(str(part) for part in item["text"] if str(part).strip()).strip()
            line = f"{'  ' * int(item['depth'])}{item['marker']} {text}".rstrip()
            rendered_item = [line, *item["children"]]
            if item_stack:
                item_stack[-1]["children"].extend(rendered_item)
            else:
                list_lines.extend(rendered_item)
        elif token_type == "table_open":
            table_rows = []
        elif token_type == "tr_open":
            row_cells = []
        elif token_type == "tr_close" and row_cells is not None:
            table_rows.append(" | ".join(row_cells))
            row_cells = None
        elif token_type == "table_close":
            content = "\n".join(table_rows)
            if content:
                blocks.append(MarkdownBlock("table", content, _section_path(headings)))
            table_rows = []
        elif token_type == "fence":
            content = token.content.strip("\n")
            if content:
                blocks.append(MarkdownBlock("code", content, _section_path(headings)))
        elif token_type == "inline":
            content = _render_inline(token)
            if content:
                if row_cells is not None:
                    row_cells.append(content)
                elif item_stack:
                    item_stack[-1]["text"].append(content)
                elif not list_stack:
                    blocks.append(MarkdownBlock("paragraph", content, _section_path(headings)))
        index += 1

    if list_lines:
        blocks.append(MarkdownBlock("list", "\n".join(list_lines), list_path))
    return ExtractedMarkdown(tuple(blocks))


Handler = Callable[[bytes, str], ExtractedMarkdown]
_HANDLERS: dict[str, Handler] = {
    ".pdf": _extract_pdf,
    ".txt": _extract_plain,
    ".md": _extract_markdown,
    ".csv": _extract_csv,
    ".docx": _extract_docx,
}


def _extract_sync(data: bytes, file_name: str) -> ExtractedMarkdown:
    suffix = PurePosixPath(file_name).suffix.lower()
    handler = _HANDLERS.get(suffix)
    if handler is None:
        raise UnsupportedMediaTypeError(
            f"cannot extract text from '{suffix or file_name}'; "
            f"supported: {', '.join(sorted(_HANDLERS))}"
        )
    return handler(data, file_name)


class TextExtractionService:
    async def extract(self, data: bytes, file_name: str) -> ExtractedMarkdown:
        return await asyncio.to_thread(_extract_sync, data, file_name)
