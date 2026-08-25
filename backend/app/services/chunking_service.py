"""Chunk simple Markdown blocks with the embedding model's exact tokenizer."""

import re
from collections.abc import Sequence

from app.config import Settings, get_settings
from app.services.ai_types import ChunkType, ExtractedMarkdown, MarkdownBlock, TextChunk
from app.services.embedding_service import FastEmbedEmbedder

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_BOUNDARY = re.compile(r"(?<=[;:,])\s+")
_URL = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)", re.IGNORECASE)
_LONG_TOKEN = re.compile(r"\S{64,}")


def _domain(url: str) -> str:
    return url.split("://", 1)[-1].split("/", 1)[0]


def normalise_for_model(content: str) -> str:
    """Keep abnormal links/tokens out of embeddings while preserving stored content."""

    def markdown_link(match: re.Match[str]) -> str:
        return f"{match.group(1)}. External link to {_domain(match.group(2))}."

    def bare_url(match: re.Match[str]) -> str:
        return f"External link to {_domain(match.group(0))}."

    normalised = _MARKDOWN_LINK.sub(markdown_link, content)
    normalised = _URL.sub(bare_url, normalised)
    return _LONG_TOKEN.sub("[unusually long token omitted]", normalised)


def build_embedding_text(document_title: str, chunk: TextChunk) -> str:
    parts = [f"Document: {document_title}"] if document_title else []
    if chunk.section_path:
        parts.append(f"Section: {chunk.section_path}")
    parts.append(f"Type: {chunk.chunk_type}")
    parts.append(normalise_for_model(chunk.content))
    return "\n".join(parts)


class ChunkingService:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: FastEmbedEmbedder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or FastEmbedEmbedder(self.settings)
        self.max_tokens = self.settings.chunk_max_tokens

    async def split(
        self,
        document: ExtractedMarkdown,
        *,
        document_title: str = "",
    ) -> list[TextChunk]:
        return self.split_sync(document, document_title=document_title)

    def split_sync(
        self,
        document: ExtractedMarkdown,
        *,
        document_title: str = "",
    ) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        pending: MarkdownBlock | None = None

        def append(block_type: ChunkType, section_path: str | None, content: str) -> None:
            if content.strip():
                chunks.append(TextChunk(len(chunks), block_type, section_path, content.strip()))

        def flush_pending() -> None:
            nonlocal pending
            if pending is not None:
                append(pending.block_type, pending.section_path, pending.content)
                pending = None

        for block in document.blocks:
            if not block.content.strip():
                continue
            if block.block_type == "paragraph":
                pieces = self._split_text(block, document_title)
                for piece in pieces:
                    if pending is None:
                        pending = MarkdownBlock("paragraph", piece, block.section_path)
                        continue
                    combined = f"{pending.content}\n\n{piece}"
                    candidate = MarkdownBlock("paragraph", combined, block.section_path)
                    if pending.section_path == block.section_path and self._fits(
                        candidate, document_title
                    ):
                        pending = candidate
                    else:
                        flush_pending()
                        pending = MarkdownBlock("paragraph", piece, block.section_path)
                continue

            flush_pending()
            if block.block_type == "list":
                pieces = self._split_lines(block, document_title)
            elif block.block_type == "table":
                pieces = self._split_table(block, document_title)
            else:
                pieces = self._split_text(block, document_title)
            for piece in pieces:
                append(block.block_type, block.section_path, piece)
        flush_pending()
        return chunks

    def _fits(self, block: MarkdownBlock, document_title: str) -> bool:
        chunk = TextChunk(0, block.block_type, block.section_path, block.content)
        return (
            self.embedder.count_tokens(build_embedding_text(document_title, chunk))
            <= self.max_tokens
        )

    def _split_text(self, block: MarkdownBlock, document_title: str) -> list[str]:
        if self._fits(block, document_title):
            return [block.content.strip()]
        sentences = [
            part.strip() for part in _SENTENCE_BOUNDARY.split(block.content) if part.strip()
        ]
        units: list[str] = []
        for sentence in sentences:
            candidate = MarkdownBlock(block.block_type, sentence, block.section_path)
            if self._fits(candidate, document_title):
                units.append(sentence)
                continue
            clauses = [part.strip() for part in _CLAUSE_BOUNDARY.split(sentence) if part.strip()]
            if len(clauses) > 1:
                units.extend(clauses)
            else:
                units.extend(self._word_windows(block, sentence, document_title))
        return self._pack_units(block, units, document_title)

    def _pack_units(
        self,
        block: MarkdownBlock,
        units: Sequence[str],
        document_title: str,
    ) -> list[str]:
        packed: list[str] = []
        current = ""
        for unit in units:
            candidate_text = f"{current} {unit}".strip()
            candidate = MarkdownBlock(block.block_type, candidate_text, block.section_path)
            if self._fits(candidate, document_title):
                current = candidate_text
                continue
            if current:
                packed.append(current)
                current = ""
            single = MarkdownBlock(block.block_type, unit, block.section_path)
            if self._fits(single, document_title):
                current = unit
            else:
                packed.extend(self._word_windows(block, unit, document_title))
        if current:
            packed.append(current)
        return packed

    def _word_windows(
        self,
        block: MarkdownBlock,
        text: str,
        document_title: str,
    ) -> list[str]:
        words = text.split()
        if not words:
            return []
        windows: list[str] = []
        start = 0
        while start < len(words):
            end = start
            best = ""
            while end < len(words):
                candidate_text = " ".join(words[start : end + 1])
                candidate = MarkdownBlock(block.block_type, candidate_text, block.section_path)
                if not self._fits(candidate, document_title):
                    break
                best = candidate_text
                end += 1
            if not best:
                best = words[start]
                end = start + 1
            windows.append(best)
            if end >= len(words):
                break
            window_words = max(1, end - start)
            overlap = max(1, window_words // 6)
            start = max(start + 1, end - overlap)
        return windows

    def _split_lines(self, block: MarkdownBlock, document_title: str) -> list[str]:
        lines = [line.strip() for line in block.content.splitlines() if line.strip()]
        packed: list[str] = []
        current: list[str] = []
        for line in lines:
            candidate = "\n".join([*current, line])
            if self._fits(MarkdownBlock("list", candidate, block.section_path), document_title):
                current.append(line)
                continue
            if current:
                packed.append("\n".join(current))
                current = []
            line_block = MarkdownBlock("list", line, block.section_path)
            if self._fits(line_block, document_title):
                current = [line]
            else:
                packed.extend(self._split_text(line_block, document_title))
        if current:
            packed.append("\n".join(current))
        return packed

    def _split_table(self, block: MarkdownBlock, document_title: str) -> list[str]:
        rows = [row.strip() for row in block.content.splitlines() if row.strip()]
        if not rows or self._fits(block, document_title):
            return [block.content.strip()] if block.content.strip() else []
        header, *body = rows
        if not body:
            return self._split_text(block, document_title)
        packed: list[str] = []
        current: list[str] = [header]
        for row in body:
            candidate = "\n".join([*current, row])
            if self._fits(MarkdownBlock("table", candidate, block.section_path), document_title):
                current.append(row)
                continue
            if len(current) > 1:
                packed.append("\n".join(current))
                current = [header]
            single = f"{header}\n{row}"
            single_block = MarkdownBlock("table", single, block.section_path)
            if self._fits(single_block, document_title):
                current = [header, row]
            else:
                packed.extend(self._split_text(single_block, document_title))
        if len(current) > 1:
            packed.append("\n".join(current))
        return packed
