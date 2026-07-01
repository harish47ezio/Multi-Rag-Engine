"""
HuggingFace-backed `BaseTokenizer`.

Delegates to the `tokenizers` library's `Tokenizer.from_pretrained(repo)`,
which downloads on first call and caches afterwards. This is the same
mechanism the previous `OllamaAdapter` used internally — extracted here
so the embedder side no longer has to know about it.
"""

import logging

from tokenizers import Tokenizer

from rag.tokenizer.base_tokenizer import BaseTokenizer

logger = logging.getLogger(__name__)


class HFTokenizer(BaseTokenizer):

    def __init__(self, repo: str):
        self.repo = repo
        logger.info("HFTokenizer init repo=%s", repo)
        self._tokenizer: Tokenizer = Tokenizer.from_pretrained(repo)
        logger.info("HFTokenizer loaded repo=%s", repo)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)
