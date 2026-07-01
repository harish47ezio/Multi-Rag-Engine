"""
Cross-encoder reranking role.

Where embedders score a query and a candidate independently (bi-encoder),
a reranker takes the `(query, candidate)` pair *together* and emits one
relevance score — strictly more accurate at the cost of one forward pass
per pair. The pipeline calls this AFTER first-stage retrieval has cut
the candidate set down to top-k, so the cost stays bounded.

Identity contract:
    Rerankers do NOT know about fingerprints, chunk_indices, or storage.
    They consume `(query, text)` pairs and return scores aligned 1:1
    with the input order. The caller (pipeline) is responsible for
    hydrating chunk text from SQLite and re-attaching `chunk_index`
    to the reranked output.

Concrete backends live alongside this file (`OllamaReranker`,
`HFReranker`) and dispatch is done by the registry-driven factory in
`rag.factory.factory`.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class BaseReranker(ABC):

    @abstractmethod
    def score(self, query: str, texts: List[str]) -> List[float]:
        """
        Score `(query, text)` pairs and return one float per text,
        aligned with the input order. Larger == more relevant.

        Score scale is backend-defined (logits, sigmoid probability,
        cosine, …); callers must NOT compare scores across rerankers.
        """
        pass

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[int, str]],
        top_n: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Score every `(chunk_index, text)` pair and return them sorted
        by score descending. `top_n` truncates to the best N (default:
        all candidates).
        """
        if not candidates:
            return []
        texts = [text for _, text in candidates]
        scores = self.score(query, texts)
        if len(scores) != len(candidates):
            raise RuntimeError(
                f"Reranker returned {len(scores)} scores for "
                f"{len(candidates)} candidates."
            )
        scored = [
            (chunk_index, float(score))
            for (chunk_index, _), score in zip(candidates, scores)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        if top_n is not None:
            scored = scored[:top_n]
        return scored

    @abstractmethod
    def model_key(self) -> str:
        """Backend-facing identifier of the served reranker model (for logs)."""
        pass
