"""The guardrail chain. Pure functions - no Docker, no database, no model."""

import pytest

from app.config import Settings
from app.services.guardrail_service import (
    GuardrailService,
    HiddenCharacterCheck,
    MaxLengthCheck,
    NotEmptyCheck,
)

SETTINGS = Settings(guardrail_max_query_chars=50)


def service(**overrides: object) -> GuardrailService:
    return GuardrailService(settings=SETTINGS.model_copy(update=overrides))


# -- the happy path --------------------------------------------------------


async def test_an_ordinary_question_passes_every_check() -> None:
    outcome = await service().run("what does the leave policy say about carry-over?")

    assert outcome.passed
    assert outcome.failure is None
    assert [verdict.name for verdict in outcome.verdicts] == [
        "not_empty",
        "max_length",
        "hidden_characters",
    ]


async def test_tabs_and_newlines_are_allowed() -> None:
    """A pasted multi-line question is normal input, not an attack."""
    outcome = await service().run("first line\n\tsecond line")
    assert outcome.passed


# -- individual checks -----------------------------------------------------


@pytest.mark.parametrize("query", ["", "   ", "\n\t  \n"])
async def test_blank_queries_are_rejected(query: str) -> None:
    outcome = await service().run(query)

    assert not outcome.passed
    failure = outcome.failure
    assert failure is not None
    assert failure.name == "not_empty"


async def test_over_length_query_is_rejected_and_the_limit_is_stated() -> None:
    outcome = await service().run("x" * 51)

    failure = outcome.failure
    assert failure is not None
    assert failure.name == "max_length"
    # the caller needs the limit to shorten and retry - it is not a secret
    assert "50" in (failure.detail or "")


async def test_a_query_exactly_at_the_limit_passes() -> None:
    assert (await service().run("x" * 50)).passed


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("zero-width space", 0x200B),
        ("zero-width non-joiner", 0x200C),
        ("zero-width joiner", 0x200D),
        ("left-to-right mark", 0x200E),
        ("word joiner", 0x2060),
        ("byte-order mark", 0xFEFF),
    ],
)
async def test_zero_width_characters_are_rejected(name: str, code: int) -> None:
    """Invisible once rendered, and a documented way to smuggle instructions."""
    # by codepoint, not by pasting the character: an invisible literal in a
    # test is unreviewable, and a stray copy-paste would silently change it
    outcome = await service().run(f"what is the{chr(code)} leave policy?")

    failure = outcome.failure
    assert failure is not None, name
    assert failure.name == "hidden_characters"


async def test_control_characters_are_rejected() -> None:
    outcome = await service().run("leave policy\x07")

    failure = outcome.failure
    assert failure is not None
    assert failure.name == "hidden_characters"


# -- chain behaviour -------------------------------------------------------


async def test_the_chain_stops_at_the_first_failure() -> None:
    """Fail-fast, so an expensive validator never sees input already refused."""
    outcome = await service().run("")

    # not_empty failed, so max_length and hidden_characters never ran
    assert [verdict.name for verdict in outcome.verdicts] == ["not_empty"]


async def test_passing_verdicts_are_kept_alongside_the_failure() -> None:
    outcome = await service().run("x" * 51)

    assert [(v.name, v.passed) for v in outcome.verdicts] == [
        ("not_empty", True),
        ("max_length", False),
    ]


async def test_hidden_character_check_can_be_switched_off() -> None:
    permissive = service(guardrail_reject_hidden_characters=False)

    assert [check.name for check in permissive.checks] == ["not_empty", "max_length"]
    assert (await permissive.run(f"what is the{chr(0x200B)} leave policy?")).passed


async def test_a_custom_chain_replaces_the_defaults_entirely() -> None:
    """The seam the Guardrails AI adapter uses: hand it a different list."""
    custom = GuardrailService(checks=[NotEmptyCheck()], settings=SETTINGS)

    assert [check.name for check in custom.checks] == ["not_empty"]
    # the length ceiling is not enforced, because that check is not in this chain
    assert (await custom.run("x" * 5000)).passed


async def test_checks_are_usable_on_their_own() -> None:
    """Each check is independently constructible - no service needed."""
    assert (await NotEmptyCheck().check("hello")).passed
    assert not (await MaxLengthCheck(3).check("hello")).passed
    assert not (await HiddenCharacterCheck().check(f"hel{chr(0x200B)}lo")).passed
