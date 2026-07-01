import logging
from typing import List, Optional

from providers.hugging_face_client import HuggingFaceClient
from rag.embedder.base_embedder import BaseEmbedder
from rag.search.distance_metrics.base_distance_metric import MetricKind

logger = logging.getLogger(__name__)

class HuggingFaceAdapter(BaseEmbedder):


    def __init__(
        self,
        client: HuggingFaceClient,
        model: str,
        metric_kind: MetricKind,
    ):
        self.client = client
        self.model = model
        self._metric_kind = metric_kind
        self._dimension: Optional[int] = None
        logger.info(
            "HuggingFaceAdapter init model=%s metric=%s",
            model,
            metric_kind.value,
        )


    def embed(self, text: List[str]) -> List[List[float]]:
        logger.info("HuggingFaceAdapter.embed inputs=%d", len(text))
        vectors = self.client.embed(self.model, text)
        if self._dimension is None and vectors:
            self._dimension = len(vectors[0])
            logger.info("HuggingFaceAdapter.embed dimension cached=%d", self._dimension)
        return vectors

    def dimension(self) -> int:
        if self._dimension is None:
            logger.info("HuggingFaceAdapter.dimension probing via dummy embed")
            self.embed(["dimension probe"])
        return self._dimension

    def recommended_metric(self) -> MetricKind:
        return self._metric_kind