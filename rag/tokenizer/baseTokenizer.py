"""
`BaseTokenizer` — sibling abstraction to `BaseEmbedder`.

Token counting is intentionally split out of the embedder contract:
the model's tokenizer is a property of the *weights*, not the
*transport*. Any provider serving the same model (Ollama, HuggingFace
local, HuggingFace API, TEI) reaches the same tokenizer; conversely,
a single provider may serve many models with different tokenizers.

Keeping `count_tokens` here means:
  * a new provider adapter (HF API, OpenAI, …) does NOT have to learn
    about tokenizers — it consumes a `BaseTokenizer` injected by the
    factory.
  * a new tokenizer kind (`tiktoken`, an API tokenizer, an approximation)
    is one new `BaseTokenizer` subclass; no provider adapter changes.

The chunker calls `instance.count_tokens(text)` which delegates here,
so the existing call sites are unchanged.
"""

from abc import ABC, abstractmethod


class BaseTokenizer(ABC):

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return the number of tokens this tokenizer would emit for `text`."""
        pass
