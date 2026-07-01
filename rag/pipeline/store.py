"""
SQLite-backed chunk store.

Chunks are scoped by `(fingerprint, model_key, chunk_index)` because
different embedding models produce different token counts, which means
different chunk boundaries — same document chunked under two models is
two independent chunk sets, neither of which should clobber the other.

The single shared `chunks.db` lives at `storage/chunks.db`. On first
call `init_db()` will create the table; if a legacy table without the
`model_key` column is detected, it is dropped and recreated — the old
contents become unreachable anyway because nothing in the new pipeline
queries by `(fingerprint, chunk_index)` alone.
"""

import logging
import sqlite3
from typing import List, Optional

from common.paths import CHUNK_DB_PATH as DB_PATH
from rag.pipeline.models import Chunk

logger = logging.getLogger(__name__)


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [row["name"] for row in rows]


def init_db() -> None:
    """
    Create the `chunks` table, or drop+recreate if a legacy schema is
    detected (i.e. missing the `model_key` column).
    """
    logger.info("init_db path=%s", DB_PATH)
    with _connect() as conn:
        existing = _table_columns(conn, "chunks")
        if existing and "model_key" not in existing:
            logger.warning(
                "init_db legacy chunks table detected (cols=%s); dropping",
                existing,
            )
            conn.execute("DROP TABLE chunks")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                fingerprint     TEXT    NOT NULL,
                model_key       TEXT    NOT NULL,
                chunk_index     INTEGER NOT NULL,
                text            TEXT    NOT NULL,
                start_page      INTEGER NOT NULL,
                end_page        INTEGER NOT NULL,
                source          TEXT    NOT NULL,
                page_char_start INTEGER NOT NULL,
                page_char_end   INTEGER NOT NULL,
                overlap_source  TEXT,
                PRIMARY KEY (fingerprint, model_key, chunk_index)
            )
            """
        )


def save_chunks(chunks: List[Chunk], model_key: str) -> None:
    """Persist chunks to SQLite scoped to (fingerprint, model_key)."""
    if not chunks:
        logger.info("save skipped (empty)")
        return
    payload = []
    for c in chunks:
        d = vars(c).copy()
        d["model_key"] = model_key
        payload.append(d)
    with _connect() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO chunks
                (fingerprint, model_key, chunk_index, text, start_page, end_page, source,
                 page_char_start, page_char_end, overlap_source)
            VALUES
                (:fingerprint, :model_key, :chunk_index, :text, :start_page, :end_page, :source,
                 :page_char_start, :page_char_end, :overlap_source)
            """,
            payload,
        )
    logger.info(
        "save_chunks saved=%d model_key=%s path=%s",
        len(chunks),
        model_key,
        DB_PATH,
    )


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    """Build a Chunk from a row, ignoring the row's `model_key` (not part of Chunk)."""
    return Chunk(
        fingerprint=row["fingerprint"],
        chunk_index=row["chunk_index"],
        text=row["text"],
        start_page=row["start_page"],
        end_page=row["end_page"],
        source=row["source"],
        page_char_start=row["page_char_start"],
        page_char_end=row["page_char_end"],
        overlap_source=row["overlap_source"],
    )


def get_chunk(
    fingerprint: str,
    model_key: str,
    chunk_index: int,
) -> Optional[Chunk]:
    """Fetch a single chunk by (fingerprint, model_key, chunk_index)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM chunks WHERE fingerprint = ? AND model_key = ? AND chunk_index = ?",
            (fingerprint, model_key, chunk_index),
        ).fetchone()
    found = row is not None
    logger.info(
        "get_chunk fp=%s model=%s idx=%d found=%s",
        fingerprint,
        model_key,
        chunk_index,
        found,
    )
    if row is None:
        return None
    return _row_to_chunk(row)


def get_multi_chunks(
    fingerprint: str,
    model_key: str,
    chunk_indices: List[int],
) -> List[Chunk]:
    """Fetch multiple chunks for one (document, model), preserving caller order."""
    if not chunk_indices:
        return []
    placeholders = ",".join("?" * len(chunk_indices))
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM chunks "
            f"WHERE fingerprint = ? AND model_key = ? "
            f"AND chunk_index IN ({placeholders})",
            (fingerprint, model_key, *chunk_indices),
        ).fetchall()
    by_index = {row["chunk_index"]: _row_to_chunk(row) for row in rows}
    result = [by_index[i] for i in chunk_indices if i in by_index]
    logger.info(
        "get_multi_chunks fp=%s model=%s requested=%d found=%d",
        fingerprint,
        model_key,
        len(chunk_indices),
        len(result),
    )
    return result


def get_all_chunks(fingerprint: str, model_key: str) -> List[Chunk]:
    """Fetch all chunks for one (document, model), ordered by chunk_index."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM chunks WHERE fingerprint = ? AND model_key = ? "
            "ORDER BY chunk_index",
            (fingerprint, model_key),
        ).fetchall()
    chunks = [_row_to_chunk(row) for row in rows]
    logger.info(
        "get_all_chunks fp=%s model=%s count=%d",
        fingerprint,
        model_key,
        len(chunks),
    )
    return chunks
