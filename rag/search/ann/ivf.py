import logging
import time
from typing import List, Tuple

import faiss

from rag.search.base_vector_search import BaseVectorSearcher
from rag.search.distance_metrics.base_distance_metric import (
    BaseDistanceMetric,
    MetricKind,
    UnsupportedMetricError,
)
from rag.search._validation import validate_index_inputs

logger = logging.getLogger(__name__)


# FAISS metric constant for each conceptual metric.
_FAISS_METRIC_BY_KIND = {
    MetricKind.COSINE:    faiss.METRIC_INNER_PRODUCT,
    MetricKind.DOT:       faiss.METRIC_INNER_PRODUCT,
    MetricKind.EUCLIDEAN: faiss.METRIC_L2,
}

# Quantizer (the index used to assign vectors to clusters) per metric.
_FAISS_QUANTIZER_BY_KIND = {
    MetricKind.COSINE:    faiss.IndexFlatIP,
    MetricKind.DOT:       faiss.IndexFlatIP,
    MetricKind.EUCLIDEAN: faiss.IndexFlatL2,
}

# How to turn the FAISS raw distance back into a similarity score (higher = better).
# Cosine/Dot already come back as similarities (IP); L2 is a distance, so invert.
_SCORE_BY_KIND = {
    MetricKind.COSINE:    lambda d: float(d),
    MetricKind.DOT:       lambda d: float(d),
    MetricKind.EUCLIDEAN: lambda d: float(1.0 / (1.0 + d)),
}


class IVFSearcher(BaseVectorSearcher):
    """
    Approximate Nearest Neighbor search using IVF (Inverted File Index) via FAISS.
    Partitions vector space into clusters at index time.
    At search time only searches n_probe nearest clusters — not all vectors.

    Best for: large corpora where HNSW graph doesn't fit in RAM.
    Tradeoff: slightly lower recall than HNSW, much lower memory.
    """

    def __init__(
        self,
        metric: BaseDistanceMetric,
        n_clusters: int = 100,
        n_probe: int = 10,
    ):
        """
        Args:
            metric      : distance metric instance (CosineMetric, EuclideanMetric, DotProductMetric)
            n_clusters  : number of Voronoi cells to partition vectors into
                          rule of thumb: sqrt(n_chunks)
                          too few = large cells = slow search
                          too many = small cells = poor recall
            n_probe     : how many nearest clusters to search at query time
                          higher = better recall, slower search
                          n_probe == n_clusters → exact KNN, defeats the purpose
        """
        self._metric = metric
        self._n_clusters = n_clusters
        self._n_probe = n_probe
        self._chunk_indices: List[int] = []
        self._index = None
        self._dim: int = None

    def index(self, vectors: List[List[float]], chunk_indices: List[int]) -> None:
        """
        Step 1: Metric normalizes vectors (cosine → normalize, others → raw)
        Step 2: K-Means clusters vectors into n_clusters Voronoi cells
        Step 3: Builds inverted list — centroid → [chunk vectors in that cell]
        """
        validate_index_inputs(vectors, chunk_indices)
        start = time.perf_counter()
        logger.info(
            "IVF index start count=%d dim=%d n_clusters=%d n_probe=%d",
            len(chunk_indices),
            len(vectors[0]),
            self._n_clusters,
            self._n_probe,
        )

        if len(chunk_indices) < self._n_clusters:
            # FAISS requires more vectors than clusters
            old = self._n_clusters
            self._n_clusters = max(1, len(chunk_indices) // 4)
            logger.info(
                "IVF n_clusters adjusted from=%d to=%d (chunks=%d)",
                old,
                self._n_clusters,
                len(chunk_indices),
            )

        self._chunk_indices = self._normalize_chunk_indices(chunk_indices)
        self._dim = len(vectors[0])

        # Let metric handle normalization decision
        matrix = self._metric.index_matrix(vectors)  # shape: (n_chunks, dim)

        # Build quantizer — compares query to centroids
        # Quantizer is itself a flat exact index over centroids only
        kind = self._metric.metric_kind()
        try:
            quantizer_cls = _FAISS_QUANTIZER_BY_KIND[kind]
            faiss_metric = _FAISS_METRIC_BY_KIND[kind]
        except KeyError as e:
            raise UnsupportedMetricError(
                f"IVFSearcher does not support metric kind {kind}."
            ) from e

        quantizer = quantizer_cls(self._dim)
        self._index = faiss.IndexIVFFlat(quantizer, self._dim, self._n_clusters, faiss_metric)

        # FAISS requires training step — this is where K-Means runs
        self._index.train(matrix)

        # Add vectors. FAISS auto-assigns sequential labels matching position
        # in _chunk_indices; the real chunk_index lookup happens in search().
        self._index.add(matrix)

        # Set n_probe — how many clusters to search at query time
        self._index.nprobe = self._n_probe
        logger.info(
            "IVF index done count=%d clusters=%d elapsed=%.2fs",
            len(self._chunk_indices),
            self._n_clusters,
            time.perf_counter() - start,
        )

    def search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Step 1: Metric normalizes query vector
        Step 2: Compare query to all centroids — pick n_probe nearest
        Step 3: Search only chunks inside those n_probe cells
        Step 4: Return top_k results
        query_text is ignored — IVF is vector-only.
        """
        if self._index is None:
            raise RuntimeError("Index is empty. Call index() before search().")

        start = time.perf_counter()
        # Metric returns a float32 ndarray of shape (dim,); FAISS needs (1, dim) batch shape
        q = self._metric.index_query(query_vector).reshape(1, -1)

        top_k = min(top_k, len(self._chunk_indices))

        # Returns (distances, indices) — shape (1, top_k) each
        distances, labels = self._index.search(q, top_k)

        kind = self._metric.metric_kind()
        try:
            to_score = _SCORE_BY_KIND[kind]
        except KeyError as e:
            raise UnsupportedMetricError(
                f"IVFSearcher does not support metric kind {kind}."
            ) from e

        results = []
        for label, distance in zip(labels[0], distances[0]):
            if label == -1:
                # FAISS returns -1 for empty slots (can happen if n_probe cells have < top_k vectors)
                continue
            results.append((self._chunk_indices[label], to_score(distance)))

        results.sort(key=lambda x: x[1], reverse=True)
        logger.info(
            "IVF search hits=%d top_k=%d elapsed=%.4fs",
            len(results),
            top_k,
            time.perf_counter() - start,
        )
        return results

    # ---------- persistence ----------

    def save(self, path: str) -> None:
        """
        Native FAISS binary at `path`, JSON sidecar at f"{path}.meta.json".
        FAISS index objects are not picklable, so write_index is the only
        sane choice here.
        """
        if self._index is None:
            raise RuntimeError("Nothing to save. Call index() before save().")

        logger.info("IVF save path=%s", path)
        faiss.write_index(self._index, path)

        meta = {
            "chunk_indices": self._chunk_indices,
            "dim": self._dim,
            "n_clusters": self._n_clusters,
            "n_probe": self._n_probe,
            "metric_kind": self._metric.metric_kind().value,
        }
        self._save_meta(path, meta)
        logger.info("IVF save done path=%s chunks=%d", path, len(self._chunk_indices))

    def load(self, path: str) -> None:
        """
        faiss.read_index reconstructs the quantizer and inverted lists.
        Caller must construct IVFSearcher with the same metric used at save.
        """
        logger.info("IVF load path=%s", path)
        meta = self._load_meta(path)

        self._verify_metric_on_load(meta["metric_kind"])

        self._chunk_indices = self._normalize_chunk_indices(meta["chunk_indices"])
        self._dim = meta["dim"]
        self._n_clusters = meta["n_clusters"]
        self._n_probe = meta["n_probe"]

        self._index = faiss.read_index(path)
        self._index.nprobe = self._n_probe
        logger.info(
            "IVF load done path=%s chunks=%d dim=%d clusters=%d",
            path,
            len(self._chunk_indices),
            self._dim,
            self._n_clusters,
        )
