# rag/search/bm25.py

import logging
import pickle
import time
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from .base_search import BaseSearcher
from ._validation import validate_text_only_inputs

logger = logging.getLogger(__name__)


class BM25Searcher(BaseSearcher):
    """
    Pure BM25 keyword search.
    No vectors used — scores by term frequency and inverse document frequency.
    Best for: exact terms, names, codes, IDs — things semantic search misses.
    """

    def __init__(self):
        self._texts: List[str] = []
        self._bm25: BM25Okapi = None

    def index(self, texts: List[str], vectors: List[List[float]]) -> None:
        """
        Tokenize texts and build BM25 index.
        vectors are accepted but ignored — BM25 is text-only.
        """
        validate_text_only_inputs(texts)
        start = time.perf_counter()
        logger.info("BM25 index start count=%d", len(texts))

        self._texts = texts
        tokenized = [text.lower().split() for text in texts]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(
            "BM25 index done count=%d elapsed=%.2fs",
            len(texts),
            time.perf_counter() - start,
        )

    def search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Return top_k chunks by BM25 score against query_text.
        query_vector is ignored — BM25 is text-only.
        """
        if self._bm25 is None:
            raise RuntimeError("Index is empty. Call index() before search().")

        if not query_text or not query_text.strip():
            raise ValueError("query_text must be non-empty for BM25 search.")

        start = time.perf_counter()
        tokenized_query = query_text.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        top_k = min(top_k, len(self._texts))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = [(self._texts[i], float(scores[i])) for i in top_indices]
        logger.info(
            "BM25 search hits=%d top_k=%d elapsed=%.4fs",
            len(results),
            top_k,
            time.perf_counter() - start,
        )
        return results

    # ---------- persistence ----------

    def save(self, path: str) -> None:
        """
        BM25Okapi is pure Python (term-frequency dicts + idf array) — single
        pickle is the right tool here.
        """
        if self._bm25 is None:
            raise RuntimeError("Nothing to save. Call index() before save().")

        logger.info("BM25 save path=%s", path)
        with open(path, "wb") as f:
            pickle.dump(
                {"texts": self._texts, "bm25": self._bm25},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("BM25 save done path=%s texts=%d", path, len(self._texts))

    def load(self, path: str) -> None:
        logger.info("BM25 load path=%s", path)
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._texts = state["texts"]
        self._bm25 = state["bm25"]
        logger.info("BM25 load done path=%s texts=%d", path, len(self._texts))