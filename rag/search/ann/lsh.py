# rag/search/ann/lsh.py

import logging
import pickle
import time
from typing import Dict, List, Tuple

import numpy as np

from rag.search.base_vector_search import BaseVectorSearcher
from rag.search.distance_metrics.base_distance_metric import (
    BaseDistanceMetric,
    MetricKind,
)
from rag.search._validation import validate_index_inputs

logger = logging.getLogger(__name__)


class LSHSearcher(BaseVectorSearcher):
    """
    Approximate Nearest Neighbor search using LSH (Locality Sensitive Hashing).
    Uses random hyperplane projections — SimHash variant for cosine similarity.

    No training required — random planes fixed at init time.
    Multiple hash tables improve recall at cost of memory.

    Best for: streaming data, fast index builds, when training budget is zero.
    Tradeoff: lower recall than HNSW and IVF.
    """

    def __init__(
        self,
        metric: BaseDistanceMetric,
        dim: int,
        n_hyperplanes: int = 16,
        n_tables: int = 5,
        seed: int = 42,
    ):
        self._metric = metric
        self._n_hyperplanes = n_hyperplanes
        self._n_tables = n_tables
        self._seed = seed
        self._dim = dim
        self._chunk_indices: List[int] = []
        self._vectors: np.ndarray = None
        self._tables: List[Dict[str, List[int]]] = []

        # Generate once at init — same seed + dim = identical planes every run
        rng = np.random.RandomState(self._seed)
        self._planes = [
            rng.randn(self._n_hyperplanes, self._dim).astype(np.float32)
            for _ in range(self._n_tables)
        ]

    def _hash_vector(self, vector: np.ndarray, table_idx: int) -> str:
        projections = self._planes[table_idx] @ vector
        bits = (projections > 0).astype(int)
        return "".join(map(str, bits))

    def index(self, vectors: List[List[float]], chunk_indices: List[int]) -> None:
        validate_index_inputs(vectors, chunk_indices)
        start = time.perf_counter()
        logger.info(
            "LSH index start count=%d dim=%d n_hyperplanes=%d n_tables=%d",
            len(chunk_indices),
            self._dim,
            self._n_hyperplanes,
            self._n_tables,
        )

        if len(vectors[0]) != self._dim:
            raise ValueError(f"Vector dim {len(vectors[0])} does not match initialised dim {self._dim}.")

        self._chunk_indices = self._normalize_chunk_indices(chunk_indices)
        matrix = self._metric.index_matrix(vectors)
        self._vectors = matrix
        self._tables = [{} for _ in range(self._n_tables)]

        # Bucket key is the POSITION in self._chunk_indices, not the chunk_index
        # itself — search() translates position back to chunk_index at return time.
        for position, vector in enumerate(matrix):
            for table_idx in range(self._n_tables):
                code = self._hash_vector(vector, table_idx)
                if code not in self._tables[table_idx]:
                    self._tables[table_idx][code] = []
                self._tables[table_idx][code].append(position)
        logger.info(
            "LSH index done count=%d elapsed=%.2fs",
            len(self._chunk_indices),
            time.perf_counter() - start,
        )

    def search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        if not self._tables:
            raise RuntimeError("Index is empty. Call index() before search().")

        start = time.perf_counter()
        q = np.array(self._metric.index_query(query_vector), dtype=np.float32)

        candidate_positions = set()
        for table_idx in range(self._n_tables):
            code = self._hash_vector(q, table_idx)
            if code in self._tables[table_idx]:
                for pos in self._tables[table_idx][code]:
                    candidate_positions.add(pos)

        if not candidate_positions:
            candidate_positions = set(range(len(self._chunk_indices)))

        candidate_list = list(candidate_positions)
        candidate_matrix = self._vectors[candidate_list]
        scores = self._metric.search_matrix(candidate_matrix, q)

        top_k = min(top_k, len(candidate_list))
        is_euclidean = self._metric.metric_kind() == MetricKind.EUCLIDEAN

        if is_euclidean:
            # search_matrix returns L2 distance (lower is better) — pick the
            # smallest distances, i.e. ascending order.
            top_positions = np.argsort(scores)[:top_k]
        else:
            # similarity metrics (cosine, dot): higher is better.
            top_positions = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in top_positions:
            position = candidate_list[i]
            score = float(scores[i])
            if is_euclidean:
                # search_matrix returns L2 distance; map to similarity in (0, 1].
                score = float(1 / (1 + score))
            results.append((self._chunk_indices[position], score))

        logger.info(
            "LSH search hits=%d top_k=%d candidates=%d elapsed=%.4fs",
            len(results),
            top_k,
            len(candidate_list),
            time.perf_counter() - start,
        )
        return results

    # ---------- persistence ----------

    def save(self, path: str) -> None:
        """
        LSH is pure Python + numpy — single pickle covers all state.
        Random planes are regenerated from `seed` at construction, but we
        still persist them so load() can verify integrity if seed changes.
        """
        if not self._tables:
            raise RuntimeError("Nothing to save. Call index() before save().")

        logger.info("LSH save path=%s", path)
        state = {
            "chunk_indices": self._chunk_indices,
            "vectors": self._vectors,
            "tables": self._tables,
            "planes": self._planes,
            "dim": self._dim,
            "n_hyperplanes": self._n_hyperplanes,
            "n_tables": self._n_tables,
            "seed": self._seed,
            "metric_kind": self._metric.metric_kind().value,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("LSH save done path=%s chunks=%d", path, len(self._chunk_indices))

    def load(self, path: str) -> None:
        """
        Restore state. Caller must construct LSHSearcher with the same metric
        used at save time (planes/seed/dim are restored from the pickle).
        """
        logger.info("LSH load path=%s", path)
        with open(path, "rb") as f:
            state = pickle.load(f)

        self._verify_metric_on_load(state["metric_kind"])

        self._chunk_indices = self._normalize_chunk_indices(state["chunk_indices"])
        self._vectors = state["vectors"]
        self._tables = state["tables"]
        self._planes = state["planes"]
        self._dim = state["dim"]
        self._n_hyperplanes = state["n_hyperplanes"]
        self._n_tables = state["n_tables"]
        self._seed = state["seed"]
        logger.info(
            "LSH load done path=%s chunks=%d dim=%d",
            path,
            len(self._chunk_indices),
            self._dim,
        )
