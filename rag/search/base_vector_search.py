"""
Shared base for vector-based searchers (HNSW, Annoy, IVF, LSH, KNN).

`BaseVectorSearcher` collects the boilerplate that every vector searcher
repeated verbatim — chunk-index normalization, the metric-mismatch guard on
load, and the JSON sidecar read/write — so each backend only implements the
parts that are genuinely backend-specific (index build, search, native
persistence). BM25 stays on `BaseSearcher` directly since it is text-only.
"""

import json
from typing import List, Sequence

from rag.search.base_search import BaseSearcher
from rag.search.distance_metrics.base_distance_metric import (
    BaseDistanceMetric,
    MetricKind,
)


class BaseVectorSearcher(BaseSearcher):
    """Base class holding state and persistence helpers shared by vector searchers.

    Subclasses must assign `self._metric` (a `BaseDistanceMetric`) in their
    `__init__` before calling any of these helpers.
    """

    _metric: BaseDistanceMetric

    @staticmethod
    def _normalize_chunk_indices(chunk_indices: Sequence[int]) -> List[int]:
        """Coerce chunk indices to a list of plain ints."""
        return [int(i) for i in chunk_indices]

    def _verify_metric_on_load(self, saved_metric_value: str) -> None:
        """Raise if the persisted metric kind differs from this instance's metric.

        Metric is behaviour, not data, so it is never restored from disk — the
        caller must construct the searcher with the same metric it was saved with.
        """
        saved_kind = MetricKind(saved_metric_value)
        if saved_kind != self._metric.metric_kind():
            raise ValueError(
                f"Metric mismatch on load: saved={saved_kind}, "
                f"current={self._metric.metric_kind()}."
            )

    @staticmethod
    def _save_meta(path: str, meta: dict) -> None:
        """Write the Python-side sidecar next to the native index at `path`."""
        with open(f"{path}.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    @staticmethod
    def _load_meta(path: str) -> dict:
        """Read the Python-side sidecar written by `_save_meta`."""
        with open(f"{path}.meta.json", "r", encoding="utf-8") as f:
            return json.load(f)
