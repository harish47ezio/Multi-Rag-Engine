import logging
import time
from typing import List, Tuple

import numpy as np

from rag.search.base_vector_search import BaseVectorSearcher
from rag.search.distance_metrics.base_distance_metric import (
    BaseDistanceMetric,
    MetricKind,
    UnsupportedMetricError,
)
from rag.search._validation import validate_index_inputs

logger = logging.getLogger(__name__)


# hnswlib space string for each conceptual metric.
# Cosine + Dot both use "ip" — Cosine is normalized upstream by the metric.
_HNSWLIB_SPACE_BY_KIND = {
    MetricKind.COSINE:    "ip",
    MetricKind.DOT:       "ip",
    MetricKind.EUCLIDEAN: "l2",
}

# Convert hnswlib raw distance back into a similarity score (higher = better).
# For "ip"  -> distance = 1 - dot_product, so similarity = 1 - distance.
# For "l2"  -> distance = squared L2, monotonic similarity = 1 / (1 + distance).
_SCORE_BY_KIND = {
    MetricKind.COSINE:    lambda d: 1.0 - float(d),
    MetricKind.DOT:       lambda d: 1.0 - float(d),
    MetricKind.EUCLIDEAN: lambda d: 1.0 / (1.0 + float(d)),
}


class ANNSearcher(BaseVectorSearcher):
    """
    Approximate Nearest Neighbor search using HNSW via hnswlib.
    Trades tiny accuracy loss for massive speed on large corpora.

    Best for: large documents, 10k+ chunks, production-scale retrieval.

    HNSW builds a layered graph at index time.
    At search time it navigates the graph — O(log n) instead of O(n).
    """

    def __init__(self, metric: BaseDistanceMetric, ef_construction: int = 200, M: int = 16, ef_search: int = 50):
        """
        Args:
            ef_construction : size of candidate list during index build.
                              Higher = better index quality, slower build.
                              Range: 100-500, default 200 is solid.

            M               : number of bidirectional links per node in graph.
                              Higher = better recall, more memory.
                              Range: 8-64, default 16 is solid.

            ef_search       : size of candidate list during search.
                              Higher = better recall, slower search.
                              Must be >= top_k. We set dynamically at search time.
        """
        self._ef_construction = ef_construction
        self._M = M
        self._ef_search = ef_search
        self._chunk_indices: List[int] = []
        self._index = None  # hnswlib index
        self._dim: int = None
        self._metric = metric

    def index(self, vectors: List[List[float]], chunk_indices: List[int]) -> None:
        """
        Build HNSW graph index over normalized vectors.
        This is the expensive step — done once at ingest time.
        """
        import hnswlib

        validate_index_inputs(vectors, chunk_indices)
        start = time.perf_counter()
        logger.info(
            "ANN index start count=%d dim=%d ef_c=%d M=%d ef_s=%d",
            len(chunk_indices),
            len(vectors[0]),
            self._ef_construction,
            self._M,
            self._ef_search,
        )

        self._chunk_indices = self._normalize_chunk_indices(chunk_indices)
        self._dim = len(vectors[0])

        matrix = self._metric.index_matrix(vectors)

        kind = self._metric.metric_kind()
        try:
            space = _HNSWLIB_SPACE_BY_KIND[kind]
        except KeyError as e:
            raise UnsupportedMetricError(
                f"ANNSearcher (hnswlib) does not support metric kind {kind}."
            ) from e

        # space="ip" = inner product (dot product on normalized vectors = cosine)
        self._index = hnswlib.Index(space=space, dim=self._dim)
        self._index.init_index(
            max_elements=len(self._chunk_indices),
            ef_construction=self._ef_construction,
            M=self._M
        )

        # Add all vectors with integer labels matching position in _chunk_indices.
        # Labels are positional; the real chunk_index lookup happens in search().
        self._index.add_items(matrix, list(range(len(self._chunk_indices))))
        self._index.set_ef(self._ef_search)
        logger.info(
            "ANN index done count=%d elapsed=%.2fs",
            len(self._chunk_indices),
            time.perf_counter() - start,
        )

    def search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 2
    ) -> List[Tuple[int, float]]:
        """
        Navigate HNSW graph to find approximate top_k neighbors.
        query_text is ignored — ANN is vector-only.
        """
        if self._index is None:
            raise RuntimeError("Index is empty. Call index() before search().")

        start = time.perf_counter()
        # Normalize query
        q = self._metric.index_query(query_vector)

        # ef must be >= top_k for valid results
        self._index.set_ef(max(self._ef_search, top_k))

        top_k = min(top_k, len(self._chunk_indices))

        # hnswlib returns (indices, distances).
        # For space="ip"  -> distance = 1 - dot_product
        # For space="l2"  -> distance = squared euclidean distance
        labels, distances = self._index.knn_query(q.reshape(1, -1), k=top_k)

        kind = self._metric.metric_kind()
        try:
            to_score = _SCORE_BY_KIND[kind]
        except KeyError as e:
            raise UnsupportedMetricError(
                f"ANNSearcher (hnswlib) does not support metric kind {kind}."
            ) from e

        results = [
            (self._chunk_indices[label], to_score(distance))
            for label, distance in zip(labels[0], distances[0])
        ]

        # Sort descending by score
        results.sort(key=lambda x: x[1], reverse=True)
        logger.info(
            "ANN search hits=%d top_k=%d elapsed=%.4fs",
            len(results),
            top_k,
            time.perf_counter() - start,
        )
        return results

    # ---------- persistence ----------

    def save(self, path: str) -> None:
        """
        Native hnswlib binary for the graph, JSON sidecar for python-side state.
        Sidecar lives at f"{path}.meta.json".
        """
        if self._index is None:
            raise RuntimeError("Nothing to save. Call index() before save().")

        logger.info("ANN save path=%s", path)
        self._index.save_index(path)

        meta = {
            "chunk_indices": self._chunk_indices,
            "dim": self._dim,
            "ef_construction": self._ef_construction,
            "M": self._M,
            "ef_search": self._ef_search,
            "metric_kind": self._metric.metric_kind().value,
        }
        self._save_meta(path, meta)
        logger.info("ANN save done path=%s chunks=%d", path, len(self._chunk_indices))

    def load(self, path: str) -> None:
        """
        Caller must construct ANNSearcher with the SAME metric that was used
        at save time. Hyperparameters (ef_*, M) are restored from the sidecar.
        """
        import hnswlib

        logger.info("ANN load path=%s", path)
        meta = self._load_meta(path)

        self._verify_metric_on_load(meta["metric_kind"])

        self._chunk_indices = self._normalize_chunk_indices(meta["chunk_indices"])
        self._dim = meta["dim"]
        self._ef_construction = meta["ef_construction"]
        self._M = meta["M"]
        self._ef_search = meta["ef_search"]

        kind = self._metric.metric_kind()
        try:
            space = _HNSWLIB_SPACE_BY_KIND[kind]
        except KeyError as e:
            raise UnsupportedMetricError(
                f"ANNSearcher (hnswlib) does not support metric kind {kind}."
            ) from e

        self._index = hnswlib.Index(space=space, dim=self._dim)
        # max_elements must be >= the count baked into the file.
        self._index.load_index(path, max_elements=len(self._chunk_indices))
        self._index.set_ef(self._ef_search)
        logger.info(
            "ANN load done path=%s chunks=%d dim=%d",
            path,
            len(self._chunk_indices),
            self._dim,
        )
