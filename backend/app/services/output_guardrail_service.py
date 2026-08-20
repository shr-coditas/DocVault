"""Checks run on generated text before anyone sees it.

All of it is mechanical - regex, unicode categories, substring overlap - and all
of it is cheap, which is the reason it is here rather than in a second model
call. Asking a model whether another model's output leaked the system prompt
costs a request, takes a second, and is less reliable at this than ``in``.

Ordering matters more than any individual rule: this runs on the draft, before
citations are resolved. Resolution drops markers that point at no supplied
source, so a check that ran after it would inspect a tidy bibliography and never
learn that the model invented ``[7]``.
"""

import re
import unicodedata
from collections.abc import Sequence

from app.config import Settings
from app.services.ai_types import OutputIssue, OutputVerdict, SearchHit

# [1], [2, 3], [4,5] - the shape the system prompt asks for. Anything else the
# model writes simply is not a citation as far as this is concerned, which is
# the safe direction to fail.
_CITATION = re.compile(r"\[((?:\d+)(?:\s*,\s*\d+)*)\]")

# Names that only appear in an answer if the model started describing its own
# configuration instead of the user's documents.
_CONFIG_DISCLOSURE = re.compile(
    r"\b(system prompt|developer message|hidden instructions|jwt_secret|"
    r"s3_secret_key|api[_ -]?key|llm_provider)\b",
    re.I,
)

# Tab, newline and carriage return are the only control characters a person
# writes. The rest are invisible once rendered.
_ALLOWED_CONTROL = frozenset({"\t", "\n", "\r"})


def check_output(
    draft: str,
    sources: Sequence[SearchHit],
    *,
    system_prompt: str,
    settings: Settings,
) -> OutputVerdict:
    """Every issue the draft has, not just the first one.

    All of them are collected because the corrective sentence sent with a retry
    is built from the list: telling the model only about the missing citations
    when it also copied a paragraph verbatim buys a second rejection.
    """
    issues: list[OutputIssue] = []

    if _overlaps(draft, system_prompt, minimum=settings.agent_output_prompt_overlap_chars):
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
        _overlaps(draft, source.content, minimum=settings.agent_output_source_overlap_chars)
        for source in sources
    ):
        issues.append(OutputIssue.EXCESSIVE_SOURCE_COPY)

    return OutputVerdict(tuple(dict.fromkeys(issues)))


def _overlaps(left: str, right: str, *, minimum: int) -> bool:
    """Whether the two share a run of at least ``minimum`` characters.

    Whitespace and case are normalized first so that reformatting a quoted
    passage does not hide it. Comparing every window of the shorter string
    against every window of the longer is quadratic in principle and irrelevant
    in practice: an answer is a few thousand characters at most.
    """
    normalized_left = " ".join(left.casefold().split())
    normalized_right = " ".join(right.casefold().split())
    if min(len(normalized_left), len(normalized_right)) < minimum:
        return False
    shorter, longer = sorted((normalized_left, normalized_right), key=len)
    windows = {shorter[start : start + minimum] for start in range(len(shorter) - minimum + 1)}
    return any(
        longer[start : start + minimum] in windows for start in range(len(longer) - minimum + 1)
    )
