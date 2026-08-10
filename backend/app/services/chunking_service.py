"""Deterministic structure-first, context-aware document chunk planning."""

import asyncio
import hashlib
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Protocol, cast

from app.config import Settings, get_settings
from app.services.ai_types import (
    ChunkSourceSpan,
    ChunkType,
    DocumentNode,
    ExtractedArtifact,
    ExtractedDocument,
    SourceLocation,
    TextChunk,
)
from app.services.text_extraction_service import artifact_from_blocks
from app.services.token_counting import ApproximateTokenCounter, TokenCounter

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_END = re.compile(r"(?<=[;:])\s+")
_WORD = re.compile(r"\S+\s*")
_CODE_BOUNDARY = re.compile(
    r"^(?:async\s+def|def|class|function|interface|type|export\s+function)\b"
)
_JOINER = "\n\n"


@dataclass(frozen=True, slots=True)
class _Piece:
    text: str
    node: DocumentNode
    locations: tuple[SourceLocation, ...]
    continuation_before: bool = False
    continuation_after: bool = False


class ChunkPlanner(Protocol):
    async def split(
        self,
        extracted: ExtractedArtifact | ExtractedDocument,
        *,
        document_title: str | None = None,
    ) -> list[TextChunk]: ...


class ChunkingService:
    """Chunk an AST using semantic boundaries before token constraints."""

    profile_name = "structure-v2"

    def __init__(
        self,
        settings: Settings | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.tokens = token_counter or ApproximateTokenCounter()

    async def split(
        self,
        extracted: ExtractedArtifact | ExtractedDocument,
        *,
        document_title: str | None = None,
    ) -> list[TextChunk]:
        return await asyncio.to_thread(self.split_sync, extracted, document_title=document_title)

    def split_sync(
        self,
        extracted: ExtractedArtifact | ExtractedDocument,
        *,
        document_title: str | None = None,
    ) -> list[TextChunk]:
        artifact = (
            artifact_from_blocks(extracted, document_title=document_title or "Document")
            if isinstance(extracted, ExtractedDocument)
            else extracted
        )
        if artifact.is_empty:
            return []
        title = document_title or str(
            artifact.metadata.get("document_title")
            or artifact.metadata.get("file_name")
            or artifact.root.text
            or "Document"
        )
        nodes = {node.id: node for node in artifact.nodes}
        children: dict[uuid.UUID, list[DocumentNode]] = defaultdict(list)
        for node in artifact.nodes:
            if node.parent_id is not None:
                children[node.parent_id].append(node)
        for rows in children.values():
            rows.sort(key=lambda node: node.ordinal)

        drafts: list[TextChunk] = []
        self._walk_container(artifact.root, nodes, children, title, drafts)
        return [replace(chunk, chunk_index=index) for index, chunk in enumerate(drafts)]

    def _walk_container(
        self,
        container: DocumentNode,
        nodes: dict[uuid.UUID, DocumentNode],
        children: dict[uuid.UUID, list[DocumentNode]],
        title: str,
        output: list[TextChunk],
    ) -> None:
        paragraph_pieces: list[_Piece] = []
        pending_caption: DocumentNode | None = None

        def flush_paragraphs() -> None:
            nonlocal paragraph_pieces
            if paragraph_pieces:
                self._pack_pieces(
                    paragraph_pieces,
                    "fallback_text_chunk"
                    if any(piece.node.attributes.get("fallback_text") for piece in paragraph_pieces)
                    else "paragraph_chunk",
                    container,
                    nodes,
                    title,
                    output,
                )
                paragraph_pieces = []

        for node in children.get(container.id, []):
            if node.node_type in {
                "heading",
                "page_break",
                "suppressed_header",
                "suppressed_footer",
            }:
                continue
            if node.node_type == "caption":
                flush_paragraphs()
                pending_caption = node
                continue
            if node.node_type == "section":
                flush_paragraphs()
                pending_caption = None
                self._walk_container(node, nodes, children, title, output)
                continue
            if node.node_type == "paragraph":
                paragraph_pieces.extend(
                    self._split_node(node, container, nodes, title, "paragraph_chunk")
                )
                continue

            if node.node_type == "list":
                introduction = paragraph_pieces[-1:] if paragraph_pieces else []
                preceding = paragraph_pieces[:-1]
                if preceding:
                    self._pack_pieces(
                        preceding,
                        "paragraph_chunk",
                        container,
                        nodes,
                        title,
                        output,
                    )
                paragraph_pieces = []
                self._chunk_list(
                    node,
                    children,
                    container,
                    nodes,
                    title,
                    output,
                    introduction=introduction,
                )
                continue

            flush_paragraphs()
            caption = pending_caption.text if pending_caption else None
            pending_caption = None
            if node.node_type == "table":
                self._chunk_table(node, children, container, nodes, title, output, caption)
            elif node.node_type == "list_item":
                self._pack_pieces(
                    self._split_node(node, container, nodes, title, "list_item_chunk"),
                    "list_item_chunk",
                    container,
                    nodes,
                    title,
                    output,
                )
            elif node.node_type == "code_block":
                self._pack_pieces(
                    self._split_code(node, container, nodes, title),
                    "code_chunk",
                    container,
                    nodes,
                    title,
                    output,
                )
            elif node.node_type in {"quote", "note", "warning", "figure"}:
                chunk_type = cast(
                    ChunkType,
                    {
                        "quote": "quote_chunk",
                        "note": "note_chunk",
                        "warning": "warning_chunk",
                        "figure": "figure_chunk",
                    }[node.node_type],
                )
                text = "\n".join(part for part in (caption, node.text) if part)
                if text:
                    contextual = replace(node, text=text)
                    self._pack_pieces(
                        self._split_node(contextual, container, nodes, title, chunk_type),
                        chunk_type,
                        container,
                        nodes,
                        title,
                        output,
                    )
            elif node.text:
                paragraph_pieces.extend(
                    self._split_node(node, container, nodes, title, "paragraph_chunk")
                )
        flush_paragraphs()

    def _chunk_list(
        self,
        node: DocumentNode,
        children: dict[uuid.UUID, list[DocumentNode]],
        container: DocumentNode,
        nodes: dict[uuid.UUID, DocumentNode],
        title: str,
        output: list[TextChunk],
        introduction: list[_Piece] | None = None,
    ) -> None:
        items = children.get(node.id, [])
        pieces = [
            _Piece(
                self._render_list_item(item, children),
                item,
                self._descendant_locations(item, children),
            )
            for item in items
            if item.node_type == "list_item" and self._render_list_item(item, children).strip()
        ]
        if not pieces and node.text:
            pieces = [_Piece(node.text, node, node.source_spans)]
        self._pack_pieces(
            [*(introduction or []), *pieces], "list_chunk", container, nodes, title, output
        )

    def _render_list_item(
        self, node: DocumentNode, children: dict[uuid.UUID, list[DocumentNode]], depth: int = 0
    ) -> str:
        own = f"{node.attributes.get('marker', '')}{node.text or ''}"
        descendants: list[str] = []
        for child in children.get(node.id, []):
            if child.node_type == "list":
                descendants.extend(
                    self._render_list_item(item, children, depth + 1)
                    for item in children.get(child.id, [])
                    if item.node_type == "list_item"
                )
        prefix = "  " * depth
        rendered = f"{prefix}{own}" if own else ""
        return "\n".join(part for part in [rendered, *descendants] if part)

    def _chunk_table(
        self,
        node: DocumentNode,
        children: dict[uuid.UUID, list[DocumentNode]],
        container: DocumentNode,
        nodes: dict[uuid.UUID, DocumentNode],
        title: str,
        output: list[TextChunk],
        caption: str | None,
    ) -> None:
        rows = [child for child in children.get(node.id, []) if child.node_type == "table_row"]
        raw_headers = node.attributes.get("headers", [])
        headers = [str(value) for value in raw_headers] if isinstance(raw_headers, list) else []
        header_line = " | ".join(headers)
        prefix = "\n".join(part for part in (caption, header_line) if part)
        if not rows:
            text = "\n".join(part for part in (prefix, node.text) if part)
            if text:
                self._pack_pieces(
                    self._split_node(
                        replace(node, text=text), container, nodes, title, "table_chunk"
                    ),
                    "table_chunk",
                    container,
                    nodes,
                    title,
                    output,
                )
            return

        rendered = [row.text or "" for row in rows if row.text]
        complete = "\n".join(part for part in (prefix, *rendered) if part)
        if (
            self._contextual_tokens(complete, container, nodes, title, "table_chunk")
            <= self.settings.chunk_target_tokens
        ):
            combined = replace(
                node, text=complete, source_spans=self._descendant_locations(node, children)
            )
            self._pack_pieces(
                [_Piece(complete, combined, combined.source_spans)],
                "table_chunk",
                container,
                nodes,
                title,
                output,
            )
            return

        row_pieces = [
            _Piece(
                "\n".join(part for part in (prefix, row.text) if part),
                row,
                row.source_spans,
            )
            for row in rows
            if row.text and not row.attributes.get("header")
        ]
        self._pack_pieces(
            row_pieces,
            "table_row_group_chunk",
            container,
            nodes,
            title,
            output,
            joiner="\n",
            deduplicate_prefix=prefix,
        )

    def _split_code(
        self,
        node: DocumentNode,
        container: DocumentNode,
        nodes: dict[uuid.UUID, DocumentNode],
        title: str,
    ) -> list[_Piece]:
        text = node.text or ""
        lines = text.splitlines(keepends=True)
        groups: list[str] = []
        current = ""
        for line in lines:
            if current and _CODE_BOUNDARY.match(line.strip()):
                groups.append(current.rstrip())
                current = ""
            current += line
        if current.strip():
            groups.append(current.rstrip())
        if len(groups) == 1:
            return self._split_node(node, container, nodes, title, "code_chunk")
        pieces = []
        for index, group in enumerate(groups):
            pieces.append(
                _Piece(
                    group,
                    node,
                    node.source_spans,
                    continuation_before=index > 0,
                    continuation_after=index < len(groups) - 1,
                )
            )
        return pieces

    def _split_node(
        self,
        node: DocumentNode,
        container: DocumentNode,
        nodes: dict[uuid.UUID, DocumentNode],
        title: str,
        chunk_type: ChunkType,
    ) -> list[_Piece]:
        text = node.text or ""
        if not text.strip():
            return []
        if (
            self._contextual_tokens(text, container, nodes, title, chunk_type)
            <= self.settings.chunk_max_tokens
        ):
            return [_Piece(text, node, node.source_spans)]

        prefix = self._context_prefix(container, nodes, title, chunk_type)
        available = max(1, self.settings.chunk_max_tokens - self.tokens.count_tokens(prefix))
        segments = self._recursive_split(text, available)
        pieces = []
        cursor = 0
        for index, segment in enumerate(segments):
            start = text.find(segment, cursor)
            if start < 0:
                start = cursor
            end = start + len(segment)
            cursor = end
            locations = tuple(
                self._slice_location(location, start, end) for location in node.source_spans
            )
            pieces.append(
                _Piece(
                    segment,
                    node,
                    locations,
                    continuation_before=index > 0,
                    continuation_after=index < len(segments) - 1,
                )
            )
        return pieces

    def _recursive_split(self, text: str, maximum: int) -> list[str]:
        if self.tokens.count_tokens(text) <= maximum:
            return [text]
        for separator in (_SENTENCE_END, _CLAUSE_END):
            parts = [part for part in separator.split(text) if part]
            if len(parts) > 1:
                return self._pack_text_parts(parts, maximum)
        words = [match.group() for match in _WORD.finditer(text)]
        if len(words) > 1:
            return self._pack_text_parts(words, maximum, joiner="")
        results = []
        cursor = 0
        while cursor < len(text):
            accepted = cursor + 1
            while (
                accepted < len(text)
                and self.tokens.count_tokens(text[cursor : accepted + 1]) <= maximum
            ):
                accepted += 1
            results.append(text[cursor:accepted])
            cursor = accepted
        return results

    def _pack_text_parts(self, parts: list[str], maximum: int, joiner: str = " ") -> list[str]:
        groups: list[str] = []
        current = ""
        for part in parts:
            candidate = f"{current}{joiner if current else ''}{part}"
            if current and self.tokens.count_tokens(candidate) > maximum:
                groups.append(current)
                current = part
            else:
                current = candidate
        if current:
            groups.append(current)
        return [subpart for group in groups for subpart in self._recursive_split(group, maximum)]

    def _pack_pieces(
        self,
        pieces: list[_Piece],
        chunk_type: ChunkType,
        container: DocumentNode,
        nodes: dict[uuid.UUID, DocumentNode],
        title: str,
        output: list[TextChunk],
        *,
        joiner: str = _JOINER,
        deduplicate_prefix: str | None = None,
    ) -> None:
        current: list[_Piece] = []
        for piece in pieces:
            candidate = [*current, piece]
            content = self._join_pieces(candidate, joiner, deduplicate_prefix)
            if (
                current
                and self._contextual_tokens(content, container, nodes, title, chunk_type)
                > self.settings.chunk_target_tokens
            ):
                output.append(
                    self._make_chunk(
                        current,
                        chunk_type,
                        container,
                        nodes,
                        title,
                        len(output),
                        joiner,
                        deduplicate_prefix,
                    )
                )
                current = []
            current.append(piece)
            content = self._join_pieces(current, joiner, deduplicate_prefix)
            if (
                self._contextual_tokens(content, container, nodes, title, chunk_type)
                > self.settings.chunk_max_tokens
            ):
                too_large = current.pop()
                if current:
                    output.append(
                        self._make_chunk(
                            current,
                            chunk_type,
                            container,
                            nodes,
                            title,
                            len(output),
                            joiner,
                            deduplicate_prefix,
                        )
                    )
                split = self._split_node(too_large.node, container, nodes, title, chunk_type)
                if len(split) == 1 and split[0].text == too_large.text:
                    current = [too_large]
                else:
                    self._pack_pieces(
                        split,
                        chunk_type,
                        container,
                        nodes,
                        title,
                        output,
                        joiner=joiner,
                        deduplicate_prefix=deduplicate_prefix,
                    )
                    current = []
        if current:
            output.append(
                self._make_chunk(
                    current,
                    chunk_type,
                    container,
                    nodes,
                    title,
                    len(output),
                    joiner,
                    deduplicate_prefix,
                )
            )

    @staticmethod
    def _join_pieces(pieces: list[_Piece], joiner: str, deduplicate_prefix: str | None) -> str:
        texts = []
        for index, piece in enumerate(pieces):
            text = piece.text
            if index and deduplicate_prefix and text.startswith(deduplicate_prefix):
                text = text[len(deduplicate_prefix) :].lstrip("\n")
            texts.append(text)
        return joiner.join(texts)

    def _make_chunk(
        self,
        pieces: list[_Piece],
        chunk_type: ChunkType,
        container: DocumentNode,
        nodes: dict[uuid.UUID, DocumentNode],
        title: str,
        index: int,
        joiner: str,
        deduplicate_prefix: str | None,
    ) -> TextChunk:
        content = self._join_pieces(pieces, joiner, deduplicate_prefix)
        heading_path = self._heading_path(container, nodes)
        breadcrumb = " > ".join(heading_path) or None
        context_prefix = self._context_prefix(container, nodes, title, chunk_type)
        embedding_text = f"{context_prefix}\n{content}" if context_prefix else content
        lexical_text = "\n".join(part for part in (title, breadcrumb, content) if part)
        spans: list[ChunkSourceSpan] = []
        cursor = 0
        for sequence, piece in enumerate(pieces):
            rendered = piece.text
            if sequence and deduplicate_prefix and rendered.startswith(deduplicate_prefix):
                rendered = rendered[len(deduplicate_prefix) :].lstrip("\n")
            if sequence:
                cursor += len(joiner)
            start = cursor
            cursor += len(rendered)
            locations = piece.locations or (SourceLocation(),)
            for location in locations:
                spans.append(ChunkSourceSpan(sequence, start, cursor, location))
        pages = [
            span.location.page_number for span in spans if span.location.page_number is not None
        ]
        first = pieces[0]
        logical_key = f"{first.node.logical_path}/{chunk_type}:{index}"
        metadata: dict[str, object] = {
            "node_ids": [str(piece.node.id) for piece in pieces],
            "continuation_before": first.continuation_before,
            "continuation_after": pieces[-1].continuation_after,
        }
        if any(piece.node.attributes.get("fallback_text") for piece in pieces):
            metadata["fallback_text"] = True
        return TextChunk(
            chunk_index=index,
            content=content,
            token_count=self.tokens.count_tokens(embedding_text),
            heading=heading_path[-1] if heading_path else None,
            source_spans=tuple(spans),
            embedding_text=embedding_text,
            lexical_text=lexical_text,
            chunk_type=chunk_type,
            logical_key=logical_key,
            structural_node_id=first.node.id,
            parent_node_id=container.id,
            ordinal_in_parent=first.node.ordinal,
            heading_path=heading_path,
            breadcrumb=breadcrumb,
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            language="und",
            metadata=metadata,
        )

    def _contextual_tokens(
        self,
        content: str,
        container: DocumentNode,
        nodes: dict[uuid.UUID, DocumentNode],
        title: str,
        chunk_type: ChunkType,
    ) -> int:
        prefix = self._context_prefix(container, nodes, title, chunk_type)
        return self.tokens.count_tokens(f"{prefix}\n{content}" if prefix else content)

    def _context_prefix(
        self,
        container: DocumentNode,
        nodes: dict[uuid.UUID, DocumentNode],
        title: str,
        chunk_type: ChunkType,
    ) -> str:
        breadcrumb = " > ".join(self._heading_path(container, nodes))
        labels = [f"Document: {title}"]
        if breadcrumb:
            labels.append(f"Section: {breadcrumb}")
        if chunk_type.startswith("table"):
            labels.append("Content type: table")
        elif chunk_type.startswith("list"):
            labels.append("Content type: list")
        elif chunk_type == "code_chunk":
            labels.append("Content type: code")
        return "\n".join(labels)

    @staticmethod
    def _heading_path(
        container: DocumentNode, nodes: dict[uuid.UUID, DocumentNode]
    ) -> tuple[str, ...]:
        path: list[str] = []
        current: DocumentNode | None = container
        while current is not None:
            if current.node_type == "section" and current.text:
                path.append(current.text)
            current = nodes.get(current.parent_id) if current.parent_id else None
        return tuple(reversed(path))

    def _descendant_locations(
        self, node: DocumentNode, children: dict[uuid.UUID, list[DocumentNode]]
    ) -> tuple[SourceLocation, ...]:
        locations = list(node.source_spans)
        for child in children.get(node.id, []):
            locations.extend(self._descendant_locations(child, children))
        return tuple(locations)

    @staticmethod
    def _slice_location(location: SourceLocation, start: int, end: int) -> SourceLocation:
        char_start = location.char_start + start if location.char_start is not None else None
        char_end = location.char_start + end if location.char_start is not None else None
        normalized_start = (
            location.normalized_char_start + start
            if location.normalized_char_start is not None
            else None
        )
        normalized_end = (
            location.normalized_char_start + end
            if location.normalized_char_start is not None
            else None
        )
        return replace(
            location,
            char_start=char_start,
            char_end=char_end,
            normalized_char_start=normalized_start,
            normalized_char_end=normalized_end,
        )
