"""
Single source of truth for on-disk locations used across the pipeline.

Every stage (parse -> chunk -> index -> search -> report) reads and writes
under these roots, keyed by the document fingerprint and embedder model_key.
Centralizing them here keeps the layout consistent and avoids the same string
literal drifting across modules.
"""

from pathlib import Path

# Top-level directories.
OUTPUT_DIR = Path("output")      # parser writes output/<fingerprint>.md
STORAGE_DIR = Path("storage")    # SQLite db + per-(fingerprint, model_key) indexes
RESULT_DIR = Path("result")      # per-query markdown reports

# SQLite chunk store (shared across all documents/models).
CHUNK_DB_PATH = STORAGE_DIR / "chunks.db"

# Per-(fingerprint, model_key) artifact filenames.
VECTORS_FILENAME = "vectors.npy"
INDEX_HNSW_FILENAME = "index_hnsw.bin"
INDEX_ANNOY_FILENAME = "index_annoy.ann"
INDEX_IVF_FILENAME = "index_ivf.faiss"
INDEX_LSH_FILENAME = "index_lsh.pkl"


def parsed_markdown_path(fingerprint: str) -> Path:
    """Path to the parser's page-marked markdown for a document."""
    return OUTPUT_DIR / f"{fingerprint}.md"


def doc_storage_dir(fingerprint: str, model_key: str) -> Path:
    """Directory holding vectors + every searcher index for one (doc, model)."""
    return STORAGE_DIR / fingerprint / model_key
