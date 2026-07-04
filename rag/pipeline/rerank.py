"""
Per-searcher cross-encoder rerank helper.

Searchers return `(chunk_index, score)` without ever looking at chunk
text — text only lives in the SQLite store at
`storage/chunks.db` ([rag/pipeline/store.py](rag/pipeline/store.py)).
Reranking is the first stage that needs the actual text, so this
module hydrates it on-demand and then forwards the `(chunk_index,
text)` pairs to `instance.rerank(...)`.

Returning `(elapsed_seconds, reranked_results)` keeps the surface
matching `run_search` so `alpha_test.py` can swap a `run_search` tuple
out for a reranked one with a single line change.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

from rag.factory.instance import EmbeddingInstance, RerankerInstance
from rag.pipeline.store import get_multi_chunks

logger = logging.getLogger(__name__)


def rerank_results(
    name: str,
    results: List[Tuple[int, float]],
    query: str,
    fingerprint: str,
    embedding: EmbeddingInstance,
    reranker: Optional[RerankerInstance],
    top_n: int,
) -> Tuple[float, List[Tuple[int, float]]]:
    """
    Rerank one searcher's top-k results in place of its bi-encoder scores.

    Args:
        name        : searcher name (for logging only).
        results     : `(chunk_index, score)` tuples from `run_search`.
        query       : raw query text — the cross-encoder needs it as-is.
        fingerprint : document fingerprint to scope the SQLite lookup.
        embedding   : the embedding instance, for the `model_key` under
                      which chunks were stored.
        reranker    : the active reranker instance, or None.
        top_n       : truncate the reranked list to the best `top_n`.

    Returns:
        `(elapsed_seconds, reranked_results)`. If `reranker` is None,
        returns `(0.0, results)` unchanged so the caller can wire this
        helper unconditionally.
    """
    if reranker is None:
        logger.info("rerank_results name=%s skipped (no reranker on mother instance)", name)
        return 0.0, results

    if not results:
        return 0.0, []

    indices = [idx for idx, _ in results]
    chunks = get_multi_chunks(fingerprint, embedding.model_key, indices)
    by_index = {c.chunk_index: c for c in chunks}

    pairs: List[Tuple[int, str]] = []
    missing: List[int] = []
    for idx in indices:
        chunk = by_index.get(idx)
        if chunk is None:
            missing.append(idx)
            continue
        pairs.append((idx, chunk.text))

    if missing:
        logger.warning(
            "rerank_results name=%s missing chunk text for indices=%s "
            "(dropping from rerank input)",
            name,
            missing,
        )

    start = time.perf_counter()
    reranked = reranker.rerank(query, pairs, top_n=top_n) or []
    elapsed = time.perf_counter() - start

    logger.info(
        "rerank_results name=%s reranker=%s inputs=%d outputs=%d elapsed=%.4fs",
        name,
        reranker.provider_id,
        len(pairs),
        len(reranked),
        elapsed,
    )
    return elapsed, reranked
