"""What is this query, and does it justify a retrieval?

Classification runs **before** the vector search, not after it. That ordering is
the point: "hello" and "ignore your instructions" both cost an embedding call and
an HNSW scan if they are classified afterwards, and both are worth exactly zero.
Only ``DOCUMENT_QUESTION`` reaches the index.
"""

import re

from app.services.ai_types import IntentJudgement, QueryIntent

# ---------------------------------------------------------------------------
# STUDY NOTE - why the default is DOCUMENT_QUESTION, and why that is not laziness
#
# The two errors are not symmetric.
#
#   false OUT_OF_SCOPE   a real question about the user's own documents is
#                        refused. The user has no recourse - rephrasing is a
#                        guessing game, and the system looks broken.
#   false DOCUMENT_QUESTION  a pointless retrieval runs, finds nothing above the
#                        score threshold, and step 7 declines to answer. Cost: one
#                        embedding and one indexed scan.
#
# The first is a broken product, the second is a rounding error. So the classifier
# only leaves DOCUMENT_QUESTION on an explicit signal, and retrieval's own score
# threshold is the real backstop for "we have nothing on this".
#
# This is also why a rule-based classifier is defensible here at all. It could
# never reliably separate "question about my documents" from "general knowledge
# question" - *the same sentence is either one depending on what the corpus
# holds.* "What is our leave policy?" is a document question in a workspace of HR
# files and unanswerable in a workspace of invoices, and no amount of pattern
# matching can see the difference. Rather than pretend otherwise, the rules
# recognise only the three cases that *are* decidable from the text alone:
# greetings, explicit non-document tasks, and instruction-override attempts.
#
# An LLM-backed classifier could do better, but this project currently has one
# concrete, deterministic classifier that works offline.
# ---------------------------------------------------------------------------

# Instruction-override phrasing. These target the *shape* of an attack - telling
# the system to disregard its rules, to reveal them, or to adopt a new persona -
# rather than any particular wording, because wording is trivially varied.
_INJECTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all)\b.{0,20}\b(instruction|prompt|rule|direction|context)",
            re.I,
        ),
        "instruction-override phrasing",
    ),
    (
        re.compile(
            r"\b(reveal|show|print|repeat|output|tell me)\b.{0,30}\b(system|initial|original|hidden)\b.{0,15}\b(prompt|instruction|message|rule)",
            re.I,
        ),
        "asks for the system prompt",
    ),
    (
        re.compile(
            r"\byou are (now|no longer)\b|\bpretend (you are|to be)\b|\bact as (if|though|a)\b",
            re.I,
        ),
        "persona override",
    ),
    (
        re.compile(r"\b(developer|debug|god|admin)\s+mode\b|\bjailbreak\b|\bDAN\b"),
        "jailbreak framing",
    ),
    (
        re.compile(
            r"\b(bypass|circumvent|disable|turn off)\b.{0,30}\b(guardrail|filter|safety|restriction|permission|security)",
            re.I,
        ),
        "asks to bypass controls",
    ),
    (
        # The tenancy-specific one: this system is multi-tenant, so "fetch me
        # documents I have no access to" is the attack that matters most here.
        #
        # A leading imperative is required, and that is the whole reason this
        # pattern is safe to have. Without it, "what does the all users' data
        # policy say?" - an ordinary question about an ordinary HR document -
        # matches and gets refused. Demanding show/list/give separates an
        # exfiltration attempt from a question that merely mentions other users.
        # The optional filler word is what lets "all *other* users" match while
        # keeping the quantifier and the noun adjacent enough to mean something.
        re.compile(
            r"\b(show|list|give|fetch|dump|get|retrieve|display)\b.{0,20}"
            r"\b(all|other|every|everyone|another)\b(\s+\w+)?\s+"
            r"(user|users|member|members|workspace|workspaces|tenant|tenants)('s)?\b"
            r".{0,30}\b(document|file|data)",
            re.I,
        ),
        "asks for another tenant's documents",
    ),
)

# Whole-query greetings and pleasantries. Anchored, because "hello, what does the
# leave policy say?" is a document question with a greeting stuck on the front -
# only a query that is *nothing but* a pleasantry is chitchat.
_CHITCHAT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(hi|hey|hello|yo|greetings|good (morning|afternoon|evening))[\s!.,?]*$", re.I),
    re.compile(r"^\s*(thanks|thank you|ta|cheers|nice|cool|ok|okay|got it)[\s!.,?]*$", re.I),
    re.compile(r"^\s*(bye|goodbye|see you|later)[\s!.,?]*$", re.I),
    re.compile(r"^\s*how are you( doing)?[\s!.,?]*$", re.I),
    re.compile(r"^\s*(who|what) are you[\s!.,?]*$", re.I),
    re.compile(r"^\s*what can you do( for me)?[\s!.,?]*$", re.I),
)

# Explicit requests to do something that is not "answer from the documents".
# Every one of these names a task, which is what makes it decidable from the text
# - unlike a topic, which is not.
_OUT_OF_SCOPE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(write|compose|draft|generate)\s+(me\s+)?(a|an|some)?\s*(poem|song|story|joke|essay|rap|haiku)\b",
            re.I,
        ),
        "asks for creative writing",
    ),
    (
        re.compile(
            r"\b(write|generate|debug|fix|refactor)\b.{0,20}\b(code|script|function|program|sql|regex)\b",
            re.I,
        ),
        "asks for code",
    ),
    (
        re.compile(
            r"\b(what|how)('s| is)? the weather\b|\bweather (in|for|today|tomorrow)\b", re.I
        ),
        "asks about the weather",
    ),
    (
        re.compile(r"\b(stock price|share price|exchange rate|bitcoin price)\b", re.I),
        "asks for live market data",
    ),
    (
        re.compile(
            r"\btranslate\b.{0,30}\b(to|into)\s+(english|french|german|spanish|hindi|japanese|chinese)\b",
            re.I,
        ),
        "asks for translation",
    ),
)


class RuleBasedIntentClassifier:
    # a matched pattern is a deliberate, specific signal
    MATCH_CONFIDENCE = 0.9
    # the default is a policy choice, not an observation. See the note above.
    DEFAULT_CONFIDENCE = 0.4

    async def classify(self, query: str) -> IntentJudgement:
        text = query.strip()

        for pattern, reason in _INJECTION_PATTERNS:
            if pattern.search(text):
                return IntentJudgement(QueryIntent.PROMPT_INJECTION, self.MATCH_CONFIDENCE, reason)

        for pattern in _CHITCHAT_PATTERNS:
            if pattern.match(text):
                return IntentJudgement(
                    QueryIntent.CHITCHAT, self.MATCH_CONFIDENCE, "greeting or pleasantry"
                )

        for pattern, reason in _OUT_OF_SCOPE_PATTERNS:
            if pattern.search(text):
                return IntentJudgement(QueryIntent.OUT_OF_SCOPE, self.MATCH_CONFIDENCE, reason)

        return IntentJudgement(
            QueryIntent.DOCUMENT_QUESTION,
            self.DEFAULT_CONFIDENCE,
            "no rule matched; treated as a document question by default",
        )
