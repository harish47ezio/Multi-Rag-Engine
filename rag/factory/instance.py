"""
The runtime objects the pipeline actually consumes.

The registry describes *what's possible*; these objects are *locked
picks* ready to run. There are four independent building blocks plus a
composite that ties them together:

  * `EmbeddingInstance` — one model + tokenizer + provider + metric.
    Composes a `BaseTokenizer` (token counting, chunker concern), a
    `BaseEmbedder` (text -> vector) and a `BaseDistanceMetric`
    (similarity scoring). Exposes `embed` / `count_tokens` / `dimension`
    / `metric` as one facade.

  * `RerankerInstance` — one cross-encoder reranker (`BaseReranker`).

  * `LLMInstance` — one text-generation model (`BaseLLMAdapter`).

  * `SearchInstance` — the chosen subset of searcher class names to run.

  * `MotherInstance` — one of each of the above (reranker + llm optional),
    the single object the pipeline is parameterised by at runtime.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from llm.base import BaseLLMAdapter, LLMResponse
from rag.embedder.base_embedder import BaseEmbedder
from rag.reranker.base_reranker import BaseReranker
from rag.search.distance_metrics.base_distance_metric import BaseDistanceMetric
from rag.tokenizer.base_tokenizer import BaseTokenizer


@dataclass
class EmbeddingInstance:
    name: Optional[str]
    template_key: str
    model_key: str
    tokenizer_id: str
    provider_id: str
    tokenizer: BaseTokenizer
    embedder: BaseEmbedder
    metric: BaseDistanceMetric

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.embed(texts)

    def count_tokens(self, text: str) -> int:
        return self.tokenizer.count_tokens(text)

    def dimension(self) -> int:
        return self.embedder.dimension()

    def describe(self) -> str:
        return (
            f"{self.template_key} / tokenizer={self.tokenizer_id} "
            f"/ provider={self.provider_id} / metric={self.metric.metric_kind().value}"
        )


@dataclass
class RerankerInstance:
    name: Optional[str]
    template_key: str
    provider_id: str
    reranker: BaseReranker

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[int, str]],
        top_n: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Forward `(chunk_index, text)` candidates through the reranker."""
        return self.reranker.rerank(query, candidates, top_n)

    def describe(self) -> str:
        return f"{self.template_key} / provider={self.provider_id}"


@dataclass
class LLMInstance:
    name: Optional[str]
    template_key: str
    provider_id: str
    llm: BaseLLMAdapter

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        return self.llm.complete(prompt, **kwargs)

    def is_available(self) -> bool:
        return self.llm.is_available()

    def describe(self) -> str:
        return f"{self.template_key} / provider={self.provider_id}"


@dataclass
class SearchInstance:
    name: Optional[str]
    strategies: List[str] = field(default_factory=list)

    def describe(self) -> str:
        return ", ".join(self.strategies) if self.strategies else "(none)"


@dataclass
class MotherInstance:
    """
    The single object the pipeline is parameterised by. `reranker` and
    `llm` are optional so a mother instance can run embedding + search
    alone; callers branch on "is it None?" with one check.
    """

    name: Optional[str]
    embedding: EmbeddingInstance
    search: SearchInstance
    reranker: Optional[RerankerInstance] = None
    llm: Optional[LLMInstance] = None

    def describe(self) -> str:
        parts = [
            f"embedding=[{self.embedding.describe()}]",
            f"search=[{self.search.describe()}]",
        ]
        parts.append(
            f"reranker=[{self.reranker.describe()}]"
            if self.reranker is not None
            else "reranker=none"
        )
        parts.append(
            f"llm=[{self.llm.describe()}]" if self.llm is not None else "llm=none"
        )
        prefix = f"{self.name}: " if self.name else ""
        return prefix + "  ".join(parts)
