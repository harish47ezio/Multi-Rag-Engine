from typing import List

import numpy as np

from rag.search.distance_metrics.base_distance_metric import (
    BaseDistanceMetric,
    MetricKind,
    ScoreType,
)


class CosineDistanceMetric(BaseDistanceMetric):
    score_type = ScoreType.SIMILARITY

    def index_matrix(self, vectors: List[List[float]]) -> np.ndarray:
        matrix = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        return matrix / norms

    def search_matrix(
        self,
        chunks_matrix: np.ndarray,
        query_vector: List[float],
    ) -> np.ndarray:
        query = self.index_query(query_vector)
        return chunks_matrix @ query


    def index_query(self, query_vector: List[float]) -> np.ndarray:
        query = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm == 0:
            raise ValueError("Query vector is zero — embedding likely failed.")
        query = query / norm
        return query

    def metric_kind(self) -> MetricKind:
        return MetricKind.COSINE