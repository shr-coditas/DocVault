"""Measure contextual follow-up resolution against the checked-in labeled set."""

import argparse
import asyncio
import json
from pathlib import Path

from app.services.contextual_followup_evaluation import (
    evaluate_followups,
    load_followup_cases,
)
from app.services.contextual_query_service import get_default_contextual_resolver

DEFAULT_GOLDEN = Path("evals/contextual_followups.json")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the configured contextual query resolver."
    )
    parser.add_argument("golden", nargs="?", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    cases = load_followup_cases(args.golden)
    report = await evaluate_followups(cases, get_default_contextual_resolver())
    # ASCII escaping keeps the intentionally embedded zero-width-character case
    # printable in Windows terminals that still default to cp1252.
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_run(_arguments())))


if __name__ == "__main__":
    main()
