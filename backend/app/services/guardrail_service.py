"""Input guardrails: cheap, mechanical checks run before anything is spent.

This is the seam the Guardrails AI adapter plugs into later. Everything here is
deliberately hand-rolled and dependency-free, for the same reason the embedder
was: a check you wrote is a check you can explain, and the library version has to
earn its place against a working baseline rather than against nothing.

Two rules shape the design.

**Cheapest first, fail fast.** The chain stops at the first failure. Today every
check is a few microseconds of string work, so the ordering saves nothing worth
measuring - but a Guardrails AI validator may call a model, and the chain must
already be shaped so that a 2 KB blob of junk is rejected on length before
anything expensive looks at it.

**A guardrail decides validity, not meaning.** Whether a query is *allowed* is
here; what it is *about* belongs to the intent classifier. Keeping that line
sharp is what stops the two from quietly duplicating each other's rules - the
failure mode being an injection attempt that one blocks, the other reclassifies,
and neither logs.
"""

import unicodedata
from typing import ClassVar

from app.config import Settings, get_settings
from app.services.ai_types import GuardrailOutcome, GuardrailVerdict


class NotEmptyCheck:
    """Reject a query with no content.

    Distinct from the router's ``min_length=1``: that catches an absent
    parameter, this catches whitespace, which would otherwise reach the embedder
    and come back as a degenerate vector with confident-looking neighbours.
    """

    name = "not_empty"

    async def check(self, query: str) -> GuardrailVerdict:
        if not query.strip():
            return GuardrailVerdict(self.name, False, "the query is empty")
        return GuardrailVerdict(self.name, True)


class MaxLengthCheck:
    """Reject a query longer than the configured ceiling."""

    name = "max_length"

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars

    async def check(self, query: str) -> GuardrailVerdict:
        if len(query) > self.max_chars:
            # the limit is not a secret - a caller needs it to shorten and retry
            return GuardrailVerdict(
                self.name, False, f"the query exceeds {self.max_chars} characters"
            )
        return GuardrailVerdict(self.name, True)


class HiddenCharacterCheck:
    """Reject zero-width and control characters.

    A user typing a question produces none of these. They are, however, a
    standard way to hide instructions inside text that looks innocent when
    rendered - a zero-width joiner between the letters of a word defeats a naive
    pattern match while leaving the text readable to a model.

    Tab, newline and carriage return are allowed: those a human does type.
    """

    name = "hidden_characters"

    # Zero-width space, ZWNJ, ZWJ, LTR/RTL marks, word joiner, BOM. Spelled as
    # codepoints rather than a regex of literal characters: those characters are
    # invisible, so a pattern containing them cannot be read, reviewed or diffed.
    _ZERO_WIDTH = frozenset(
        chr(code) for code in (0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2060, 0xFEFF)
    )
    _ALLOWED_CONTROL: ClassVar[frozenset[str]] = frozenset({"\t", "\n", "\r"})

    async def check(self, query: str) -> GuardrailVerdict:
        if any(character in self._ZERO_WIDTH for character in query):
            return GuardrailVerdict(self.name, False, "the query contains zero-width characters")
        # category Cc is control, Cf is format (soft hyphen, bidi overrides) -
        # both are invisible once rendered
        if any(
            unicodedata.category(character) in {"Cc", "Cf"}
            and character not in self._ALLOWED_CONTROL
            for character in query
        ):
            return GuardrailVerdict(self.name, False, "the query contains control characters")
        return GuardrailVerdict(self.name, True)


type InputCheck = NotEmptyCheck | MaxLengthCheck | HiddenCharacterCheck


class GuardrailService:
    def __init__(
        self,
        checks: list[InputCheck] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.checks = checks if checks is not None else self._default_checks()

    def _default_checks(self) -> list[InputCheck]:
        """Ordered cheapest-and-most-decisive first. See the module docstring."""
        checks: list[InputCheck] = [
            NotEmptyCheck(),
            MaxLengthCheck(self.settings.guardrail_max_query_chars),
        ]
        if self.settings.guardrail_reject_hidden_characters:
            checks.append(HiddenCharacterCheck())
        return checks

    async def run(self, query: str) -> GuardrailOutcome:
        """Run the chain, stopping at the first failure.

        The passing verdicts are kept alongside the failure: "blocked by
        max_length, having passed not_empty" is explainable, and a bare "blocked"
        is not.
        """
        verdicts: list[GuardrailVerdict] = []
        for check in self.checks:
            verdict = await check.check(query)
            verdicts.append(verdict)
            if not verdict.passed:
                break
        return GuardrailOutcome(verdicts=tuple(verdicts))
