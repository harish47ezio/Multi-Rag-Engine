"""
HuggingFace cross-encoder reranker.

Delegates to `sentence_transformers.CrossEncoder`, which loads model
weights from the HuggingFace hub on first call and caches them locally.
Import is deferred to `__init__` so importing `rag.reranker` for an
Ollama-only setup does NOT pay the `torch` import cost.

Cross-encoder models published for retrieval (e.g.
`BAAI/bge-reranker-v2-m3`, `cross-encoder/ms-marco-MiniLM-L-6-v2`)
expose a `.predict(pairs)` method that returns a relevance score per
`(query, passage)` pair — the contract `BaseReranker.score` expects.
"""

import logging
from typing import List

from rag.reranker.base_reranker import BaseReranker

logger = logging.getLogger(__name__)


class HFReranker(BaseReranker):

    def __init__(self, repo: str, batch_size: int = 32):
        # Deferred import keeps torch off the critical path for users
        # whose chosen registry pick has no HF reranker.
        from sentence_transformers import CrossEncoder

        self.repo = repo
        self.batch_size = batch_size
        logger.info("HFReranker init repo=%s batch_size=%d", repo, batch_size)
        self._model = CrossEncoder(repo)
        logger.info("HFReranker loaded repo=%s", repo)

    def score(self, query: str, texts: List[str]) -> List[float]:
        if not texts:
            return []
        pairs = [[query, text] for text in texts]
        logger.info(
            "HFReranker.score repo=%s pairs=%d batch=%d",
            self.repo,
            len(pairs),
            self.batch_size,
        )
        scores = self._model.predict(pairs, batch_size=self.batch_size)
        return [float(s) for s in scores]

    def model_key(self) -> str:
        return self.repo
