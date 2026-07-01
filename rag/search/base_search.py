# rag/search/base.py

from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseSearcher(ABC):
    """
    Abstract base for all search strategies.
    Every searcher must implement search(), index(), save() and load().
    Retriever only ever calls search() — never touches strategy-specific code.

    Identity contract:
        Searchers do NOT know about fingerprints or chunk text. They are scoped
        to a single document by the caller, and only track `chunk_indices`
        (per-document chunk_index ints) parallel to the indexed vectors.
        search() returns `(chunk_index, score)`; the caller knows which
        document/fingerprint the searcher belongs to from where it loaded the
        files (typically `storage/{fingerprint}/...`).

    Persistence contract:
        save(path) writes the index state to disk. Backends are free to
            also write sidecar files derived from `path` (e.g. `path.meta.json`)
            to hold Python-side state like chunk_indices and hyperparameters.
        load(path) restores state INTO this instance. The caller is
            responsible for constructing the searcher with the same metric
            it was saved with — metric is behavior, not data, and is not
            persisted.
    """

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Search for the most relevant chunks.

        Args:
            query_vector : embedded query vector (for vector-based strategies)
            query_text   : raw query string (for BM25 / keyword strategies)
            top_k        : number of results to return

        Returns:
            List of (chunk_index, score) tuples, sorted descending by score.
        """
        pass

    @abstractmethod
    def index(
        self,
        vectors: List[List[float]],
        chunk_indices: List[int],
    ) -> None:
        """
        Build or update the search index.

        Args:
            vectors       : embedding vectors, one per chunk
            chunk_indices : the per-document chunk_index of each vector,
                            aligned 1:1 with vectors
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Persist this searcher's state to `path` (and any backend-specific
        sidecar files derived from it). Must be called after index().
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Restore previously-saved state from `path` into this instance.
        After load(), search() is callable without first calling index().
        """
        pass
