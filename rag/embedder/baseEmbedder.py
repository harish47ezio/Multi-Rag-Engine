from abc import ABC, abstractmethod
from typing import List

from rag.search.distance_metrics.base_distance_metric import MetricKind


class BaseEmbedder(ABC):
    """
    Provider-side abstraction: turns text into vectors.

    Token counting is NOT on this contract — it lives on `BaseTokenizer`
    because tokenizer identity follows the model weights, not the
    transport. An `Instance` combines a `BaseEmbedder` with a
    `BaseTokenizer` (and a metric) so the pipeline can call
    `instance.embed(...)`, `instance.count_tokens(...)`, and
    `instance.metric` against one object.
    """

    @abstractmethod
    def embed(self, text: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def recommended_metric(self) -> MetricKind:
        """
        Return the conceptual distance metric this embedder's model was
        trained against. The pipeline consumes this so the indexer and
        the searchers never have to guess (and can never silently
        disagree with the embedder's training objective).
        """
        pass
