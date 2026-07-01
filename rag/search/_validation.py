# rag/search/_validation.py

"""
Shared input validation for searcher `index()` methods.

Kept as plain functions (not on BaseSearcher) so each searcher opts in explicitly
and the abstract base stays a pure contract.
"""

from typing import Sequence


def validate_index_inputs(
    vectors: Sequence[Sequence[float]],
    chunk_indices: Sequence[int],
) -> None:
    """
    Validate inputs to a vector-based searcher's index() method.

    Args:
        vectors       : embedding vectors, one per chunk.
        chunk_indices : the per-document chunk_index of each vector,
                        aligned 1:1 with vectors.

    Raises:
        ValueError: if any check fails.
    """
    if not vectors:
        raise ValueError("vectors must be non-empty.")

    if not chunk_indices:
        raise ValueError("chunk_indices must be non-empty.")

    if len(vectors) != len(chunk_indices):
        raise ValueError(
            f"vectors ({len(vectors)}) and chunk_indices ({len(chunk_indices)}) must match."
        )


def validate_text_only_inputs(texts: Sequence[str]) -> None:
    """
    Validate inputs to a text-only searcher's index() method (e.g. BM25).
    """
    if not texts:
        raise ValueError("texts must be non-empty.")
