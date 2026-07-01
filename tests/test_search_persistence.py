"""
Minimal regression tests for the shared vector-searcher persistence helpers.

Covers only the numpy-only KNN path (no native ANN backends required) so this
runs in any environment. Exercises the `BaseVectorSearcher` helpers extracted in
the cleanup: chunk-index normalization, JSON sidecar round-trip, and the
metric-mismatch guard on load.

Runnable two ways:
    pytest tests/test_search_persistence.py
    python tests/test_search_persistence.py
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.search.knn.knn_searcher import KNNSearcher
from rag.search.distance_metrics.cosine import CosineDistanceMetric
from rag.search.distance_metrics.euclidean import EuclideanDistanceMetric


def _sample_index() -> KNNSearcher:
    searcher = KNNSearcher(CosineDistanceMetric())
    searcher.index([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [10, 11, 12])
    return searcher


def test_search_returns_best_first_with_chunk_indices():
    searcher = _sample_index()
    hits = searcher.search([1.0, 0.0], "q", top_k=2)
    assert [idx for idx, _ in hits] == [10, 12]
    assert hits[0][1] >= hits[1][1]


def test_save_load_roundtrip_preserves_results():
    searcher = _sample_index()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "knn_index")
        searcher.save(path)
        assert os.path.exists(f"{path}.meta.json")

        restored = KNNSearcher(CosineDistanceMetric())
        restored.load(path)
        assert restored.search([1.0, 0.0], "q", 2) == searcher.search([1.0, 0.0], "q", 2)


def test_metric_mismatch_on_load_raises():
    searcher = _sample_index()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "knn_index")
        searcher.save(path)

        wrong_metric = KNNSearcher(EuclideanDistanceMetric())
        try:
            wrong_metric.load(path)
        except ValueError as exc:
            assert "Metric mismatch" in str(exc)
        else:
            raise AssertionError("expected ValueError on metric mismatch")


if __name__ == "__main__":
    test_search_returns_best_first_with_chunk_indices()
    test_save_load_roundtrip_preserves_results()
    test_metric_mismatch_on_load_raises()
    print("all search persistence tests passed")
