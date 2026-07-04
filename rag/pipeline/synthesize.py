"""
LLM answer synthesis over retrieved chunks.

The final RAG stage: take a searcher's top results, hydrate their text
from the SQLite store (searchers only ever return `(chunk_index,
score)` — text lives in `storage/chunks.db`, see
[rag/pipeline/store.py](rag/pipeline/store.py)), stitch them into a
grounded prompt, and ask the LLM to answer the query from that context
alone.

Mirrors the decoupled shape of `rerank_results`: it takes the specific
pieces it needs — the `LLMInstance`, a results list, and the
`EmbeddingInstance` for the `model_key` under which chunks were stored —
rather than the whole `MotherInstance`. It no-ops (returns `None`) when
`llm` is None or no context text could be hydrated, so callers can wire
it unconditionally.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from llm.base import LLMResponse
from rag.factory.instance import EmbeddingInstance, LLMInstance
from rag.pipeline.store import get_multi_chunks

logger = logging.getLogger(__name__)


def synthesize_answer(
    results: List[Tuple[int, float]],
    query: str,
    fingerprint: str,
    embedding: EmbeddingInstance,
    llm: Optional[LLMInstance],
    top_n: Optional[int] = None,
) -> Optional[LLMResponse]:
    """
    Generate an answer to `query` grounded in the given searcher results.

    Args:
        results     : `(chunk_index, score)` tuples from `run_search` /
                      `rerank_results`.
        query       : raw query text.
        fingerprint : document fingerprint to scope the SQLite lookup.
        embedding   : the embedding instance, for the `model_key` under
                      which chunks were stored.
        llm         : the active LLM instance, or None.
        top_n       : cap the number of context chunks used (best-scored
                      first, in the order `results` arrives). None uses all.

    Returns:
        The `LLMResponse`, or `None` when `llm` is None or no context
        text could be hydrated.
    """
    if llm is None:
        logger.info("synthesize_answer skipped (no llm on mother instance)")
        return None

    if not results:
        logger.info("synthesize_answer skipped (no results)")
        return None

    indices = [idx for idx, _ in results]
    if top_n is not None:
        indices = indices[:top_n]

    chunks = get_multi_chunks(fingerprint, embedding.model_key, indices)
    by_index = {c.chunk_index: c for c in chunks}
    context = "\n\n".join(by_index[idx].text for idx in indices if idx in by_index)

    if not context:
        logger.info("synthesize_answer skipped (no context text hydrated)")
        return None

    prompt = (
        "Answer the question using only the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    logger.info(
        "synthesize_answer llm=%s context_chunks=%d",
        llm.provider_id,
        sum(1 for idx in indices if idx in by_index),
    )
    return llm.complete(prompt)
