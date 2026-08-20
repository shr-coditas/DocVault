"""Deterministic checks for generated text before persistence or delivery."""

import re
import unicodedata
from collections.abc import Sequence
from typing import Protocol

from app.config import Settings, get_settings
from app.services.ai_types import OutputIssue, OutputVerdict, SearchHit

_CITATION = re.compile(r"\[((?:\d+)(?:\s*,\s*\d+)*)\]")
_CONFIG_DISCLOSURE = re.compile(
    r"\b(system prompt|developer message|hidden instructions|jwt_secret|"
    r"s3_secret_key|api[_ -]?key|llm_provider)\b",
    re.I,
)
_ALLOWED_CONTROL = frozenset({"\t", "\n", "\r"})


class OutputGuardrail(Protocol):
    async def validate(
        self,
        draft: str,
        sources: Sequence[SearchHit],
        *,
        system_prompt: str,
    ) -> OutputVerdict: ...


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _has_overlap(left: str, right: str, *, minimum_chars: int) -> bool:
    """Method to check if two strings have overlapping substrings of a minimum length.
    Args:
        left (str): The first string to compare.
        right (str): The second string to compare.
        minimum_chars (int): The minimum length of the overlapping substring to consider.
    Returns:
        bool: True if there is an overlapping substring of at least minimum_chars length, False otherwise.
    """
    normalized_left = _normalized(left)
    normalized_right = _normalized(right)
    if min(len(normalized_left), len(normalized_right)) < minimum_chars:
        return False
    shorter, longer = sorted((normalized_left, normalized_right), key=len)
    windows = {
        shorter[start : start + minimum_chars] for start in range(len(shorter) - minimum_chars + 1)
    }
    return any(
        longer[start : start + minimum_chars] in windows
        for start in range(len(longer) - minimum_chars + 1)
    )


class DeterministicOutputGuardrail:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def validate(
        self,
        draft: str,
        sources: Sequence[SearchHit],
        *,
        system_prompt: str,
    ) -> OutputVerdict:
        issues: list[OutputIssue] = []
        if _has_overlap(
            draft,
            system_prompt,
            minimum_chars=self.settings.agent_output_prompt_overlap_chars,
        ):
            issues.append(OutputIssue.SYSTEM_PROMPT_DISCLOSURE)
        if _CONFIG_DISCLOSURE.search(draft):
            issues.append(OutputIssue.CONFIGURATION_DISCLOSURE)
        if any(
            unicodedata.category(character) in {"Cc", "Cf"} and character not in _ALLOWED_CONTROL
            for character in draft
        ):
            issues.append(OutputIssue.HIDDEN_CHARACTERS)

        markers = [
            int(number)
            for match in _CITATION.finditer(draft)
            for number in re.split(r"\s*,\s*", match.group(1))
        ]
        if markers and any(marker < 1 or marker > len(sources) for marker in markers):
            issues.append(OutputIssue.UNRESOLVABLE_CITATIONS)
        if sources and not markers:
            issues.append(OutputIssue.MISSING_CITATIONS)
        if any(
            _has_overlap(
                draft,
                source.content,
                minimum_chars=self.settings.agent_output_source_overlap_chars,
            )
            for source in sources
        ):
            issues.append(OutputIssue.EXCESSIVE_SOURCE_COPY)
        return OutputVerdict(tuple(dict.fromkeys(issues)))
