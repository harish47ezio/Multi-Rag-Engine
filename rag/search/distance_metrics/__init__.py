"""
Distance-metric package.

Exposes the metric ABC + concrete implementations, plus a small
`build_metric(MetricKind | str)` factory so callers that only know the
*kind* (e.g. the embedder, or a YAML config) can get a ready-to-use
metric instance without importing concrete classes themselves.
"""

from typing import Union

from rag.search.distance_metrics.base_distance_metric import (
    BaseDistanceMetric,
    MetricKind,
    ScoreType,
    UnsupportedMetricError,
)
from rag.search.distance_metrics.cosine import CosineDistanceMetric
from rag.search.distance_metrics.dot import DotDistanceMetric
from rag.search.distance_metrics.euclidean import EuclideanDistanceMetric

_BUILDERS = {
    MetricKind.COSINE: CosineDistanceMetric,
    MetricKind.DOT: DotDistanceMetric,
    MetricKind.EUCLIDEAN: EuclideanDistanceMetric,
}


def build_metric(kind: Union[MetricKind, str]) -> BaseDistanceMetric:
    """Resolve a `MetricKind` (or its string value) to a fresh metric instance."""
    if isinstance(kind, str):
        try:
            kind = MetricKind(kind)
        except ValueError as exc:
            raise UnsupportedMetricError(
                f"Unknown metric '{kind}'. "
                f"Choose from: {[m.value for m in MetricKind]}"
            ) from exc
    cls = _BUILDERS.get(kind)
    if cls is None:
        raise UnsupportedMetricError(f"No builder registered for metric kind {kind}")
    return cls()


__all__ = [
    "BaseDistanceMetric",
    "MetricKind",
    "ScoreType",
    "UnsupportedMetricError",
    "CosineDistanceMetric",
    "DotDistanceMetric",
    "EuclideanDistanceMetric",
    "build_metric",
]
