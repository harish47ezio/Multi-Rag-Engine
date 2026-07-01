import json
import logging
import time
from typing import List, Tuple

import numpy as np
from annoy import AnnoyIndex

from rag.search.base_search import BaseSearcher
from rag.search.distance_metrics.base_distance_metric import (
    BaseDistanceMetric,
    MetricKind,
    UnsupportedMetricError,
)
from rag.search._validation import validate_index_inputs

logger = logging.getLogger(__name__)


# Annoy metric string per conceptual metric.
# "dot" requires annoy >= 1.16.
_ANNOY_METRIC_BY_KIND = {
    MetricKind.COSINE:    "angular",
    MetricKind.DOT:       "dot",
    MetricKind.EUCLIDEAN: "euclidean",
}

# Convert Annoy raw distance into a similarity score (higher = better).
# angular   -> sqrt(2 - 2*cos), invert via 1 - d^2/2.
# dot       -> returns -dot_product, so flip the sign.
# euclidean -> monotonic similarity 1 / (1 + d).
_SCORE_BY_KIND = {
    MetricKind.COSINE:    lambda d: 1.0 - (float(d) ** 2) / 2.0,
    MetricKind.DOT:       lambda d: -float(d),
    MetricKind.EUCLIDEAN: lambda d: 1.0 / (1.0 + float(d)),
}


class AnnoySearcher(BaseSearcher):
    """
    Approximate Nearest Neighbor search using Annoy (Spotify).
    Builds a forest of random projection trees at index time.
    Searches by traversing trees — O(n_trees × log n) per query.

    Best for: static corpus, low RAM, cross-platform.
    Tradeoff: index is immutable after build — no updates.
    """

    def __init__(
        self,
        metric: BaseDistanceMetric,
        dim: int,
        n_trees: int = 10,
    ):
        """
        Args:
            metric  : distance metric instance for normalization + scope
            dim     : embedding dimension — must match embedding model output
                      qwen3-embedding:8b → 4096
            n_trees : number of random projection trees in the forest
                      more trees = better recall, more memory, slower build
                      rule of thumb: 10-50 for most corpora
        """
        self._metric = metric
        self._dim = dim
        self._n_trees = n_trees
        self._chunk_indices: List[int] = []
        self._index = None

        kind = metric.metric_kind()
        try:
            self._annoy_metric = _ANNOY_METRIC_BY_KIND[kind]
        except KeyError as e:
            raise UnsupportedMetricError(
                f"AnnoySearcher does not support metric kind {kind}."
            ) from e

    def index(self, vectors: List[List[float]], chunk_indices: List[int]) -> None:
        """
        Step 1: Metric normalizes vectors
        Step 2: Add all vectors to Annoy index
        Step 3: Build forest of n_trees random projection trees
        Index is immutable after build() — no updates possible.
        """
        validate_index_inputs(vectors, chunk_indices)
        start = time.perf_counter()
        logger.info(
            "Annoy index start count=%d dim=%d n_trees=%d metric=%s",
            len(chunk_indices),
            self._dim,
            self._n_trees,
            self._annoy_metric,
        )

        self._chunk_indices = [int(i) for i in chunk_indices]
        matrix = self._metric.index_matrix(vectors)  # normalize or raw per metric

        self._index = AnnoyIndex(self._dim, self._annoy_metric)

        # Annoy labels are positional; the real chunk_index lookup happens in search().
        for i, vector in enumerate(matrix):
            self._index.add_item(i, vector.tolist())

        # Build trees — expensive step, done once
        self._index.build(self._n_trees)
        logger.info(
            "Annoy index done count=%d elapsed=%.2fs",
            len(self._chunk_indices),
            time.perf_counter() - start,
        )

    def search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Step 1: Normalize query via metric
        Step 2: Traverse all trees to collect candidates
        Step 3: Return top_k with scores
        query_text is ignored — Annoy is vector-only.
        """
        if self._index is None:
            raise RuntimeError("Index is empty. Call index() before search().")

        start = time.perf_counter()
        q = np.array(self._metric.index_query(query_vector), dtype=np.float32)

        top_k = min(top_k, len(self._chunk_indices))

        # Returns (indices, distances)
        labels, distances = self._index.get_nns_by_vector(
            q.tolist(),
            top_k,
            include_distances=True
        )

        kind = self._metric.metric_kind()
        try:
            to_score = _SCORE_BY_KIND[kind]
        except KeyError as e:
            raise UnsupportedMetricError(
                f"AnnoySearcher does not support metric kind {kind}."
            ) from e

        results = [
            (self._chunk_indices[label], to_score(distance))
            for label, distance in zip(labels, distances)
        ]

        results.sort(key=lambda x: x[1], reverse=True)
        logger.info(
            "Annoy search hits=%d top_k=%d elapsed=%.4fs",
            len(results),
            top_k,
            time.perf_counter() - start,
        )
        return results

    # ---------- persistence ----------

    def save(self, path: str) -> None:
        """
        Native Annoy binary at `path`, JSON sidecar at f"{path}.meta.json".
        Annoy requires the index to have been built (we do that in index()).
        """
        if self._index is None:
            raise RuntimeError("Nothing to save. Call index() before save().")

        logger.info("Annoy save path=%s", path)
        self._index.save(path)

        meta = {
            "chunk_indices": self._chunk_indices,
            "dim": self._dim,
            "n_trees": self._n_trees,
            "metric_kind": self._metric.metric_kind().value,
        }
        with open(f"{path}.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        logger.info("Annoy save done path=%s chunks=%d", path, len(self._chunk_indices))

    def load(self, path: str) -> None:
        """
        Annoy requires a freshly-constructed AnnoyIndex of the exact same dim
        and metric, then load() populates it.
        """
        logger.info("Annoy load path=%s", path)
        with open(f"{path}.meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        saved_kind = MetricKind(meta["metric_kind"])
        if saved_kind != self._metric.metric_kind():
            raise ValueError(
                f"Metric mismatch on load: saved={saved_kind}, "
                f"current={self._metric.metric_kind()}."
            )
        if meta["dim"] != self._dim:
            raise ValueError(
                f"Dim mismatch on load: saved={meta['dim']}, current={self._dim}."
            )

        self._chunk_indices = [int(i) for i in meta["chunk_indices"]]
        self._n_trees = meta["n_trees"]

        self._index = AnnoyIndex(self._dim, self._annoy_metric)
        self._index.load(path)
        logger.info(
            "Annoy load done path=%s chunks=%d dim=%d",
            path,
            len(self._chunk_indices),
            self._dim,
        )
