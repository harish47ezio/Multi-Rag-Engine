from typing import List

import numpy as np

from rag.search.distance_metrics.base_distance_metric import (
    BaseDistanceMetric,
    MetricKind,
    ScoreType,
)


class DotDistanceMetric(BaseDistanceMetric):
    score_type = ScoreType.SIMILARITY

    def index_matrix(self, vectors: List[List[float]]) -> np.ndarray:
        return np.array(vectors, dtype=np.float32)

    def search_matrix(
        self,
        chunks_matrix: np.ndarray,
        query_vector: List[float],
    ) -> np.ndarray:
        query = self.index_query(query_vector)
        return chunks_matrix @ query

    def index_query(self, query_vector: List[float]) -> np.ndarray:
        return np.array(query_vector, dtype=np.float32)

    def metric_kind(self) -> MetricKind:
        return MetricKind.DOT