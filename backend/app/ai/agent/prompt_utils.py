import json
from collections.abc import Sequence

from langchain_core.messages import AnyMessage

from app.services.ai_types import SearchHit

SUPERVISOR_PROMPT = """
## Role
You are **DocVault Supervisor**, responsible for guiding one turn of the DocVault pipeline.

## Task
Decide the next step (`search`, `answer`, `clarify`, `refuse`, `unsupported`) when a user asks about documents in their workspace.

## Instructions
- Never answer the question directly or perform searches yourself.
- Choose only one of the following steps:
  - **search** → run searches you list, then ask again
  - **answer** → write a cited answer from sources already retrieved
  - **clarify** → ask the user when the question is ambiguous
  - **refuse** → if the message is not about workspace documents
  - **unsupported** → if sources were retrieved but don't answer
- Use history only to turn follow-ups into standalone questions.
- Prefer answering when possible; search only if a named part is missing.
- Clarify rather than guess when references have multiple meanings.
"""


def make_decision_payload(
    question: str,
    history: Sequence[AnyMessage],
    sources: Sequence[SearchHit],
    *,
    searches_run: int,
    searches_left: int,
    already_searched: Sequence[str] = (),
    rejected_issues: Sequence[str] = (),
) -> str:
    """Everything the supervisor gets to see, as one JSON document.

    JSON rather than prose so the boundary between our framing and the user's
    text is unambiguous: a question containing "### new instructions" is a
    string value, and reads as one.
    """
    brief = {
        "question": question,
        "history": [{"role": message.type, "content": str(message.content)} for message in history],
        "searches_run": searches_run,
        "searches_left": searches_left,
        "already_searched": list(already_searched),
        "previous_answer_rejected_for": list(rejected_issues),
        "sources": [
            {
                "number": number,
                "document": hit.document_title,
                "section": hit.section_path,
                "excerpt": hit.content.strip(),
            }
            for number, hit in enumerate(sources, start=1)
        ],
    }
    return json.dumps(brief, ensure_ascii=False)
