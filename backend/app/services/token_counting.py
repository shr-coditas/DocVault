"""Model-neutral token-counting seam used by the chunker."""

from typing import Protocol


class TokenCounter(Protocol):
    def count_tokens(self, text: str) -> int: ...


class ApproximateTokenCounter:
    """Cheap fallback for isolated tests; production uses the embedder tokenizer."""

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


class GenerationTokenCounter(Protocol):
    def count_tokens(self, text: str) -> int: ...


class ConservativeGenerationTokenCounter:
    """Provider-neutral preflight estimate; deliberately errs on the high side."""

    def count_tokens(self, text: str) -> int:
        return max(1, (len(text) + 2) // 3)
