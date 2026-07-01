"""
Build a human-readable markdown report from a set of searcher runs.

Each call writes one file at `result/{fingerprint}_{timestamp}.md` containing
a timing summary followed by per-searcher result sections, with each chunk
hydrated to its page range, source, and full text via the SQLite store.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from rag.pipeline.chunker import compute_page_raw_starts
from rag.pipeline.models import Chunk
from rag.pipeline.store import get_multi_chunks

logger = logging.getLogger(__name__)

RESULT_DIR = Path("result")

# `run_search` returns (elapsed, results); results are (chunk_index, score).
SearchHit = Tuple[int, float]
Run = Tuple[str, float, List[SearchHit]]  # (name, elapsed_seconds, hits)


def write_results_md(
    query: str,
    fingerprint: str,
    model_key: str,
    runs: Sequence[Run],
    top_k: int,
    reranker_label: Optional[str] = None,
) -> Path:
    """
    Write a markdown report for one query across multiple searchers.

    Args:
        query          : the natural-language query string.
        fingerprint    : document fingerprint the searchers were loaded from.
        model_key      : embedder model_key, used to scope the chunk lookup
                         (chunks are stored per (fingerprint, model_key)).
        runs           : list of (searcher_name, elapsed_seconds, hits) tuples.
        top_k          : the top_k that was requested at search time (for header).
        reranker_label : optional identifier of the active reranker. When
                         present, the report annotates the header and the
                         per-result score column as cross-encoder scores
                         instead of bi-encoder similarities.

    Returns:
        Path to the written markdown file.
    """
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = RESULT_DIR / f"{fingerprint}_{model_key}_{timestamp}.md"

    # Chunk indices are stable across searchers, so hydrate the union once
    # and reuse the lookup map for every run.
    unique_indices = sorted({idx for _, _, hits in runs for idx, _ in hits})
    chunks = get_multi_chunks(fingerprint, model_key, unique_indices)
    by_index = {c.chunk_index: c for c in chunks}

    hydrated_by_run: List[Tuple[str, float, List[Tuple[Chunk, float]]]] = []
    for name, elapsed, hits in runs:
        hydrated = [
            (by_index[idx], score)
            for idx, score in hits
            if idx in by_index
        ]
        hydrated_by_run.append((name, elapsed, hydrated))

    # Per-source raw text + page-number -> raw-content-start dict, computed
    # lazily and reused across every chunk that shares the same source file.
    source_cache: Dict[str, Optional[Tuple[str, Dict[int, int]]]] = {}

    score_label = "reranked score" if reranker_label else "score"

    lines: List[str] = []
    lines.append("# Search Results")
    lines.append("")
    lines.append(f"**Query:** {query}")
    lines.append(f"**Fingerprint:** `{fingerprint}`")
    lines.append(f"**Model:** `{model_key}`")
    if reranker_label:
        lines.append(f"**Reranker:** `{reranker_label}` (cross-encoder)")
    lines.append(f"**Top-k:** {top_k}")
    lines.append(f"**Generated:** {timestamp}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Searcher | Elapsed |")
    lines.append("|----------|---------|")
    for name, elapsed, _ in hydrated_by_run:
        lines.append(f"| {name} | {elapsed:.4f}s |")
    lines.append("")

    for name, elapsed, hydrated in hydrated_by_run:
        lines.append("---")
        lines.append("")
        lines.append(f"## {name} Result")
        lines.append("")
        lines.append(f"**top_k = {top_k}** &nbsp;|&nbsp; **elapsed = {elapsed:.4f}s** &nbsp;|&nbsp; **hits = {len(hydrated)}**")
        lines.append("")

        if not hydrated:
            lines.append("_No results._")
            lines.append("")
            continue

        for rank, (chunk, score) in enumerate(hydrated, start=1):
            page_range = (
                str(chunk.start_page)
                if chunk.start_page == chunk.end_page
                else f"{chunk.start_page} – {chunk.end_page}"
            )
            lines.append(
                f"### {rank}. Chunk #{chunk.chunk_index}  ({score_label}: {score:.4f})"
            )
            lines.append("")
            lines.append(f"- **Pages:** {page_range}")
            lines.append(f"- **Source:** `{chunk.source}`")
            if chunk.overlap_source:
                lines.append(f"- **Overlap source:** `{chunk.overlap_source}`")
            lines.append("")
            snippet = _raw_snippet_for(chunk, source_cache)
            lines.append(_blockquote(snippet if snippet is not None else chunk.text))
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(
        "report written path=%s runs=%d query='%s'",
        out_path,
        len(runs),
        query,
    )
    return out_path


def _blockquote(text: str) -> str:
    """Wrap multi-line text as a markdown blockquote."""
    stripped = text.strip()
    if not stripped:
        return "> _(empty)_"
    return "\n".join(f"> {line}" if line else ">" for line in stripped.splitlines())


def _raw_snippet_for(
    chunk: Chunk,
    source_cache: Dict[str, Optional[Tuple[str, Dict[int, int]]]],
) -> Optional[str]:
    """
    Return the chunk's text as it appears in the raw `output/<fp>.md`
    (preserving paragraphs, dialogue line breaks, and any `<!-- page: N -->`
    markers that fall inside the chunk's span). Falls back to None if the
    source file can't be read, letting the caller use the flattened
    `chunk.text` instead.

    Resolution is a single slice: page-relative offsets are converted to
    absolute raw offsets by adding the page's content-start (looked up via
    `compute_page_raw_starts`). Any page markers between `start_page` and
    `end_page` fall inside that slice for free.
    """
    src = chunk.source
    if src not in source_cache:
        try:
            raw = Path(src).read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("raw-source unavailable path=%s err=%s", src, exc)
            source_cache[src] = None
        else:
            source_cache[src] = (raw, compute_page_raw_starts(raw))

    entry = source_cache[src]
    if entry is None:
        return None

    raw, page_starts = entry
    raw_len = len(raw)

    abs_start = page_starts.get(chunk.start_page, 0) + max(0, chunk.page_char_start)
    abs_end = page_starts.get(chunk.end_page, 0) + max(0, chunk.page_char_end)
    abs_start = min(abs_start, raw_len)
    abs_end = max(abs_start, min(abs_end, raw_len))
    return raw[abs_start:abs_end]
