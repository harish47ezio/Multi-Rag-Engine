from abc import ABC, abstractmethod
from enum import Enum
from typing import List

import numpy as np


class ScoreType(Enum):
    """How to interpret the raw scores a metric returns."""
    SIMILARITY = "similarity"
    DISTANCE = "distance"


class MetricKind(Enum):
    """
    Conceptual identity of a distance metric, independent of any backend.

    Each ANN searcher owns its own mapping from MetricKind to the string,
    constant, or constructor argument its backing library expects (e.g.
    hnswlib space, FAISS metric type, Annoy metric name).
    """
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"


class UnsupportedMetricError(ValueError):
    """Raised by a searcher when it cannot honor the requested MetricKind."""
    pass


class BaseDistanceMetric(ABC):
    score_type: ScoreType

    @abstractmethod
    def index_matrix(self, vectors: List[List[float]]) -> np.ndarray:
        #Indexes chunk returns Cosine:Normalised Dot:Same as Vector, Euclidean:Same as Vector
        pass

    @abstractmethod
    def search_matrix(
        self,
        chunks_matrix: np.ndarray,
        query_vector: List[float],
    ) -> np.ndarray:
        #Searches chunk returns Cosine: Dot Product, Dot:Dot Product, Euclidean:Euclidean Distance
        pass

    @abstractmethod
    def index_query(self, query_vector: List[float]) -> np.ndarray:
        #Indexes query returns Cosine:Normalised Dot:Same as Vector, Euclidean:Same as Vector
        pass

    @abstractmethod
    def metric_kind(self) -> MetricKind:
        """Return the conceptual identity of this metric. Each searcher maps
        this to its own backend-specific value."""
        pass