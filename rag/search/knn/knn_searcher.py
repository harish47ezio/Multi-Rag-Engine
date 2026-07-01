import json
import logging
import time
from typing import List, Optional, Tuple

import numpy as np

from rag.search.base_search import BaseSearcher
from rag.search._validation import validate_index_inputs
from rag.search.distance_metrics.base_distance_metric import (
    BaseDistanceMetric,
    MetricKind,
    ScoreType,
)

logger = logging.getLogger(__name__)


class KnnSearcher(BaseSearcher):
    """
    Generic k-nearest-neighbour searcher.
    The distance metric is injected at construction time so that the
    indexed matrix and the query-time scoring rule can never disagree.
    """

    def __init__(self, metric: BaseDistanceMetric):
        if metric is None:
            raise ValueError("metric must be provided")
        self._metric = metric
        self._chunk_indices: List[int] = []
        self._matrix: Optional[np.ndarray] = None

    def index(self, vectors: List[List[float]], chunk_indices: List[int]) -> None:
        validate_index_inputs(vectors, chunk_indices)
        start = time.perf_counter()
        logger.info(
            "KNN index start count=%d dim=%d",
            len(chunk_indices),
            len(vectors[0]),
        )

        self._chunk_indices = [int(i) for i in chunk_indices]
        self._matrix = self._metric.index_matrix(vectors)
        logger.info(
            "KNN index done count=%d elapsed=%.2fs",
            len(self._chunk_indices),
            time.perf_counter() - start,
        )

    def search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 2,
    ) -> List[Tuple[int, float]]:
        if self._matrix is None:
            raise RuntimeError("Index is empty. Call index() before search().")

        start = time.perf_counter()
        scores = self._metric.search_matrix(self._matrix, query_vector)
        results = self._fetch_top_k(scores, self._metric.score_type, top_k)
        logger.info(
            "KNN search hits=%d top_k=%d elapsed=%.4fs",
            len(results),
            top_k,
            time.perf_counter() - start,
        )
        return results

    def _fetch_top_k(
        self,
        scores: np.ndarray,
        score_type: ScoreType,
        top_k: int,
    ) -> List[Tuple[int, float]]:
        top_k = min(top_k, len(self._chunk_indices))
        scores_arr = np.asarray(scores)

        # SIMILARITY: higher is better, sort descending.
        # DISTANCE:   lower is better, sort ascending.
        if score_type == ScoreType.SIMILARITY:
            order = np.argsort(-scores_arr)[:top_k]
        else:
            order = np.argsort(scores_arr)[:top_k]

        return [
            (self._chunk_indices[i], float(scores_arr[i]))
            for i in order
        ]

    # ---------- persistence ----------

    def save(self, path: str) -> None:
        """
        Matrix as .npy at `path` (numpy native — compact, fast, mmap-able),
        chunk_indices + metric_kind in JSON sidecar at f"{path}.meta.json".
        """
        if self._matrix is None:
            raise RuntimeError("Nothing to save. Call index() before save().")

        logger.info("KNN save path=%s", path)
        np.save(path, self._matrix, allow_pickle=False)

        meta = {
            "chunk_indices": self._chunk_indices,
            "metric_kind": self._metric.metric_kind().value,
        }
        with open(f"{path}.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        logger.info("KNN save done path=%s chunks=%d", path, len(self._chunk_indices))

    def load(self, path: str) -> None:
        logger.info("KNN load path=%s", path)
        with open(f"{path}.meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        saved_kind = MetricKind(meta["metric_kind"])
        if saved_kind != self._metric.metric_kind():
            raise ValueError(
                f"Metric mismatch on load: saved={saved_kind}, "
                f"current={self._metric.metric_kind()}."
            )

        self._chunk_indices = [int(i) for i in meta["chunk_indices"]]
        # np.save appends .npy if absent; np.load tolerates both forms.
        self._matrix = np.load(path if path.endswith(".npy") else f"{path}.npy",
                               allow_pickle=False)
        logger.info(
            "KNN load done path=%s chunks=%d shape=%s",
            path,
            len(self._chunk_indices),
            self._matrix.shape,
        )
