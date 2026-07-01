import logging
from typing import List, Optional

from providers.ollama_client import OllamaClient
from rag.embedder.base_embedder import BaseEmbedder
from rag.search.distance_metrics.base_distance_metric import MetricKind

logger = logging.getLogger(__name__)


class OllamaAdapter(BaseEmbedder):
    """
    Embedding-role adapter for Ollama (local or cloud — same HTTP API).

    Purely transport: knows how to call Ollama's `/api/embed` and
    declares which metric the served model was trained against. Token
    counting is the responsibility of a `BaseTokenizer` injected at the
    Instance level — this class does not load or hold a tokenizer.

    The `metric_kind` argument is *declared*, not inferred — the
    registry knows which similarity objective the model was trained
    against and passes it in at construction time. This is what lets
    the indexer and every searcher agree on metric without ever
    hard-coding cosine.
    """

    def __init__(
        self,
        client: OllamaClient,
        model: str,
        metric_kind: MetricKind,
    ):
        self.client = client
        self.model = model
        self._metric_kind = metric_kind
        self._dimension: Optional[int] = None
        logger.info(
            "OllamaAdapter init model=%s metric=%s",
            model,
            metric_kind.value,
        )

    def embed(self, text: List[str]) -> List[List[float]]:
        logger.info("OllamaAdapter.embed inputs=%d", len(text))
        vectors = self.client.embed(self.model, text)
        if self._dimension is None and vectors:
            self._dimension = len(vectors[0])
            logger.info("OllamaAdapter.embed dimension cached=%d", self._dimension)
        return vectors

    def dimension(self) -> int:
        if self._dimension is None:
            logger.info("OllamaAdapter.dimension probing via dummy embed")
            self.embed(["dimension probe"])
        return self._dimension

    def recommended_metric(self) -> MetricKind:
        return self._metric_kind
