"""
Token-aware paragraph- and sentence-aware chunker.

Pipeline (per chunking_strategy.md):
  1. Read output/<fingerprint>.md (already normalised by the parser so any
     run of newlines is collapsed to a single '\n').
  2. Strip <!-- page: N --> markers while building a page-offset map.
       Case A (marker between completed sentences) -> keep a single '\n'
       Case B (marker mid-sentence)                -> replace with a space
  3. Token-aware primary split (ladder):
        paragraph boundary -> largest sentence-aligned cut (4/3/2/1) ->
        hard token cut.
  4. Token-aware overlap (ladder, from chunk tail):
        last full paragraph -> last 4 / 3 / 2 / 1 sentences.
  5. Final clean: '\n' -> ' ', collapse runs of spaces.
  6. Emit Chunk objects with start_page/end_page derived from the offset map.

`count_tokens` is injected by the caller so the chunker never hard-codes a
tokenizer (the strategy doc requires the embedder's own tokenizer).
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import bisect
import logging
import re
import time

from rag.pipeline.models import Chunk, ChunkConfig
from rag.pipeline.store import get_all_chunks, save_chunks

logger = logging.getLogger(__name__)


# ─────────────────────────────── CONSTANTS ──────────────────────────────────

PAGE_MARKER_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")
_PAGE_MARKER_RE = PAGE_MARKER_RE  # internal alias, kept for readability below
_TERMINAL_PUNCT_RE = re.compile(r"[.!?][\"')\]]?\s*$")
_WHITESPACE = frozenset(" \t\n\r")


# ─────────────────────────── PUBLIC PAGE HELPERS ────────────────────────────

def compute_page_raw_starts(raw: str) -> Dict[int, int]:
    """
    For each `<!-- page: N -->` marker in `raw`, return N -> raw offset of
    the first non-whitespace byte of that page's content.

    This is the single source of truth used by both the chunker (at index
    time, to compute page-relative offsets) and the report (at render time,
    to resolve a chunk's `page_char_start` / `page_char_end` back to
    absolute slice positions in the same raw file).
    """
    out: Dict[int, int] = {}
    for m in PAGE_MARKER_RE.finditer(raw):
        start = m.end()
        while start < len(raw) and raw[start] in _WHITESPACE:
            start += 1
        out[int(m.group(1))] = start
    return out


# ──────────────────────────────── PUBLIC API ────────────────────────────────

def chunk(
    fingerprint: str,
    count_tokens: Callable[[str], int],
    model_key: str,
    config: Optional[ChunkConfig] = None,
) -> List[Chunk]:
    """
    Build token-aware chunks from a parsed document identified by `fingerprint`.

    Args:
        fingerprint:   Hash returned by the parser; resolves to output/<fp>.md.
        count_tokens:  Callable returning the embedder's token count for a
                       string. e.g. `instance.count_tokens`.
        model_key:     Embedding model's canonical key. Drives the storage
                       scope for chunks because different models produce
                       different token boundaries.
        config:        Optional ChunkConfig (defaults: 512 tokens, 15% overlap).

    Returns:
        List of Chunk objects ready to be embedded. `Chunk.text` is already
        cleaned (no '\n', no double spaces).
    """
    config = config or ChunkConfig()
    start_clock = time.perf_counter()
    logger.info(
        "chunk start fingerprint=%s model_key=%s chunk_tokens=%d overlap_pct=%.2f",
        fingerprint,
        model_key,
        config.chunk_tokens,
        config.overlap_pct,
    )

    source_path = Path("output") / f"{fingerprint}.md"
    if not source_path.exists():
        raise FileNotFoundError(f"Parsed document not found: {source_path}")

    chunks: List[Chunk] = []
    chunks = get_all_chunks(fingerprint, model_key)
    if chunks is not None and len(chunks) > 0:
        logger.info("chunk loaded chunks=%d model_key=%s", len(chunks), model_key)
        return chunks

    chunks.clear()
    
    raw = source_path.read_text(encoding="utf-8")
    logger.info("got the file path=%s", source_path)

    cleaned, page_offsets = _strip_page_markers(raw)
    if not cleaned:
        logger.info("chunk aborted reason=empty_after_marker_strip")
        return []

    # page -> (cleaned_page_start, raw_page_start). Used to convert a chunk's
    # cleaned offsets into page-relative raw offsets at emit time. If the file
    # had no page markers at all, treat the whole document as page 1 at offset 0.
    page_lookup: Dict[int, Tuple[int, int]] = {
        p: (c, r) for c, p, r in page_offsets
    } or {1: (0, 0)}

    paragraph_positions = [i for i, c in enumerate(cleaned) if c == "\n"]
    sentence_spans = _segment_sentences(cleaned)
    logger.info(
        "analyzed document pages=%d paragraphs=%d sentences=%d words=%d chars=%d",
        len(page_offsets),
        len(paragraph_positions),
        len(sentence_spans),
        len(cleaned.split()),
        len(cleaned),
    )

    logger.info("whole doc tokens=%d", count_tokens(cleaned))
    logger.info("started chunking")

    pos = 0
    chunk_index = 0
    overlap_source_for_current: Optional[str] = None

    while pos < len(cleaned):
        end, split_kind = _find_chunk_end(
            cleaned,
            pos,
            config.chunk_tokens,
            count_tokens,
            paragraph_positions,
            sentence_spans,
        )

        if end <= pos:
            end = min(pos + 1, len(cleaned))
        if end <= pos:
            break

        raw_span = cleaned[pos:end]
        final_text = _final_clean(raw_span)
        if not final_text:
            pos = end
            continue

        start_page = _page_at_offset(pos, page_offsets)
        end_page = _page_at_offset(max(end - 1, pos), page_offsets)
        token_count = count_tokens(final_text)

        start_cleaned_start, _ = page_lookup.get(start_page, (0, 0))
        end_cleaned_start, _ = page_lookup.get(end_page, (0, 0))
        page_char_start = pos - start_cleaned_start
        page_char_end = end - end_cleaned_start

        chunks.append(
            Chunk(
                fingerprint=fingerprint,
                chunk_index=chunk_index,
                text=final_text,
                start_page=start_page,
                end_page=end_page,
                source=str(source_path),
                page_char_start=page_char_start,
                page_char_end=page_char_end,
                overlap_source=overlap_source_for_current,
            )
        )
        # logger.info(
        #     "chunk emit idx=%d pages=%d-%d tokens=%d budget=%d fits=%s",
        #     chunk_index,
        #     start_page,
        #     end_page,
        #     token_count,
        #     config.chunk_tokens,
        #     "yes" if token_count <= config.chunk_tokens else "no",
        # )
        chunk_index += 1

        if end >= len(cleaned):
            break

        overlap_budget = max(1, int(config.chunk_tokens * config.overlap_pct))
        overlap_start, overlap_source_for_current = _find_overlap_start(
            cleaned,
            pos,
            end,
            overlap_budget,
            count_tokens,
            paragraph_positions,
            sentence_spans,
        )

        if overlap_start <= pos or overlap_start >= end:
            overlap_start = end
            overlap_source_for_current = None

        pos = overlap_start

    logger.info(
        "chunk done fingerprint=%s model_key=%s chunks=%d elapsed=%.2fs",
        fingerprint,
        model_key,
        len(chunks),
        time.perf_counter() - start_clock,
    )
    save_chunks(chunks, model_key)
    return chunks


# ───────────────────────── PAGE-MARKER STRIPPING ────────────────────────────

def _strip_page_markers(raw: str) -> Tuple[str, List[Tuple[int, int, int]]]:
    """
    Remove every <!-- page: N --> marker.

    Returns (cleaned_text, page_offsets) where each entry is the triple
        (cleaned_page_start, page_num, raw_page_start)
    marking the first character of page `page_num` in the cleaned stream
    AND in the raw stream. `raw_page_start` points at the first
    non-whitespace byte after the marker's closing `-->`, i.e. the byte
    that becomes cleaned[cleaned_page_start].

    Case A — text before marker ends in terminal punctuation: keep a single
             '\n' separator so the chunker sees a paragraph boundary.
    Case B — text before marker does NOT end in terminal punctuation: insert
             a single space so the chunker treats the seam as continuous prose.
    """
    out_parts: List[str] = []
    out_len = 0
    page_offsets: List[Tuple[int, int, int]] = []
    pos = 0

    for m in _PAGE_MARKER_RE.finditer(raw):
        page_num = int(m.group(1))

        segment = raw[pos:m.start()]
        out_parts.append(segment)
        out_len += len(segment)

        joined = "".join(out_parts)
        stripped = joined.rstrip()
        prefix_ends_sentence = bool(_TERMINAL_PUNCT_RE.search(stripped))

        out_parts = [stripped]
        out_len = len(stripped)

        pos = m.end()
        while pos < len(raw) and raw[pos] in _WHITESPACE:
            pos += 1
        more_content_follows = pos < len(raw)
        raw_page_start = pos

        if not stripped or not more_content_follows:
            page_offsets.append((out_len, page_num, raw_page_start))
            continue

        separator = "\n" if prefix_ends_sentence else " "
        out_parts.append(separator)
        out_len += 1
        page_offsets.append((out_len, page_num, raw_page_start))

    if pos < len(raw):
        out_parts.append(raw[pos:])

    return "".join(out_parts), page_offsets


def _page_at_offset(offset: int, page_offsets: List[Tuple[int, int, int]]) -> int:
    """Return the page number active at character `offset` in the cleaned text."""
    if not page_offsets:
        return 1
    starts = [s for s, _, _ in page_offsets]
    idx = bisect.bisect_right(starts, offset) - 1
    if idx < 0:
        return page_offsets[0][1]
    return page_offsets[idx][1]


# ────────────────────────── SENTENCE SEGMENTATION ───────────────────────────

def _segment_sentences(text: str) -> List[Tuple[int, int]]:
    """Return (start, end) char spans for each sentence in `text` (via pysbd)."""
    import pysbd

    seg = pysbd.Segmenter(language="en", clean=False, char_span=True)
    spans = seg.segment(text)
    return [(s.start, s.end) for s in spans]


# ─────────────────────────── PRIMARY-SPLIT LADDER ───────────────────────────

def _find_chunk_end(
    text: str,
    start: int,
    budget_tokens: int,
    count_tokens: Callable[[str], int],
    paragraph_positions: List[int],
    sentence_spans: List[Tuple[int, int]],
) -> Tuple[int, str]:
    """
    Return (end_offset, kind) for the chunk starting at `start` honouring
    the token budget. `kind` ∈ {"tail", "paragraph", "sentence", "hard"}.

    Ladder: whole-tail-fits -> paragraph -> sentence -> hard cut.
    """
    text_len = len(text)

    tail_tokens = count_tokens(text[start:])
    fits_tail = tail_tokens <= budget_tokens
    # logger.info(
    #     "ladder tail tokens=%d budget=%d fits=%s",
    #     tail_tokens, budget_tokens, "true" if fits_tail else "false",
    # )
    if fits_tail:
        return text_len, "tail"

    para_candidates = _candidates_in_range(paragraph_positions, start, text_len)
    best_para = _largest_fitting_end(
        text, start, para_candidates, budget_tokens, count_tokens,
    )
    if best_para is not None:
        return best_para, "paragraph"

    sentence_end_candidates = [
        e for s, e in sentence_spans if s >= start and e <= text_len and e > start
    ]
    best_sent = _largest_fitting_end(
        text, start, sentence_end_candidates, budget_tokens, count_tokens,
    )
    if best_sent is not None:
        return best_sent, "sentence"

    lo, hi = start + 1, text_len
    best_hard = start + 1
    while lo <= hi:
        mid = (lo + hi) // 2
        tk = count_tokens(text[start:mid])
        if tk <= budget_tokens:
            best_hard = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best_hard, "hard"


def _candidates_in_range(positions: List[int], lo: int, hi: int) -> List[int]:
    """Return positions p with lo < p < hi, sorted ascending."""
    left = bisect.bisect_right(positions, lo)
    right = bisect.bisect_left(positions, hi)
    return positions[left:right]


def _largest_fitting_end(
    text: str,
    start: int,
    end_candidates: List[int],
    budget_tokens: int,
    count_tokens: Callable[[str], int],
) -> Optional[int]:
    """
    Binary search for the largest end in `end_candidates` such that
    count_tokens(text[start:end]) <= budget_tokens. Returns None if none fit.
    Relies on the fact that token count is monotonic in end-offset.
    """
    if not end_candidates:
        return None

    lo, hi = 0, len(end_candidates) - 1
    best: Optional[int] = None
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = end_candidates[mid]
        if count_tokens(text[start:cand]) <= budget_tokens:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# ─────────────────────────────── OVERLAP ────────────────────────────────────

def _find_overlap_start(
    text: str,
    chunk_start: int,
    chunk_end: int,
    overlap_budget_tokens: int,
    count_tokens: Callable[[str], int],
    paragraph_positions: List[int],
    sentence_spans: List[Tuple[int, int]],
) -> Tuple[int, Optional[str]]:
    """
    Pick where the overlap region begins, walking from chunk_end backward.

    Priority: last full paragraph -> last 4 / 3 / 2 / 1 sentences.
    Returns (overlap_start, kind). kind is None when nothing fits the budget.
    """
    paras_in_chunk = _candidates_in_range(paragraph_positions, chunk_start, chunk_end)
    if paras_in_chunk:
        last_para_start = paras_in_chunk[-1] + 1
        if last_para_start < chunk_end:
            if count_tokens(text[last_para_start:chunk_end]) <= overlap_budget_tokens:
                return last_para_start, "paragraph"

    sents_in_chunk = [
        (s, e) for s, e in sentence_spans if s >= chunk_start and e <= chunk_end
    ]
    for n in (4, 3, 2, 1):
        if len(sents_in_chunk) < n:
            continue
        overlap_start = sents_in_chunk[-n][0]
        if overlap_start <= chunk_start or overlap_start >= chunk_end:
            continue
        if count_tokens(text[overlap_start:chunk_end]) <= overlap_budget_tokens:
            return overlap_start, f"sent:{n}"

    return chunk_end, None


# ─────────────────────────────── FINAL CLEAN ────────────────────────────────

def _final_clean(text: str) -> str:
    """Collapse internal '\n' to ' ' and squash repeated spaces."""
    text = text.replace("\n", " ")
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
