"""
The `Instance` is what the pipeline actually consumes.

A `Template` (in the registry) describes *what's possible* for a model;
an `Instance` is *one locked pick*: this model, this tokenizer, this
provider, this metric. Once you have an `Instance`, the chunker has a
tokenizer to call, the indexer has an embedder to call and a metric to
write into searchers, and the searcher loader has a `model_key` to
scope storage paths under.

Internally an Instance composes three orthogonal pieces:

  * a `BaseTokenizer`  — token counting (chunker concern)
  * a `BaseEmbedder`   — text → vector (indexer / search concern)
  * a `BaseDistanceMetric` — similarity scoring (indexer / search concern)

The Instance exposes `embed` / `count_tokens` / `dimension` / `metric`
as a single facade so the pipeline stages stay simple — they consume
one object and never have to reach into its parts.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from rag.embedder.baseEmbedder import BaseEmbedder
from rag.reranker.baseReranker import BaseReranker
from rag.search.distance_metrics.base_distance_metric import BaseDistanceMetric
from rag.tokenizer.baseTokenizer import BaseTokenizer


@dataclass
class Instance:
    template_key: str
    model_key: str
    tokenizer_id: str
    provider_id: str
    tokenizer: BaseTokenizer
    embedder: BaseEmbedder
    metric: BaseDistanceMetric
    reranker_id: Optional[str] = None
    reranker: Optional[BaseReranker] = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.embed(texts)

    def count_tokens(self, text: str) -> int:
        return self.tokenizer.count_tokens(text)

    def dimension(self) -> int:
        return self.embedder.dimension()

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[int, str]],
        top_n: Optional[int] = None,
    ) -> Optional[List[Tuple[int, float]]]:
        """
        Forward `(chunk_index, text)` candidates through the attached
        reranker. Returns `None` when the instance has no reranker so
        callers can branch on "do we rerank?" with a single check.
        """
        if self.reranker is None:
            return None
        return self.reranker.rerank(query, candidates, top_n)

    def describe(self) -> str:
        base = (
            f"{self.template_key} / tokenizer={self.tokenizer_id} "
            f"/ provider={self.provider_id} / metric={self.metric.metric_kind().value}"
        )
        if self.reranker_id:
            base += f" / reranker={self.reranker_id}"
        return base
