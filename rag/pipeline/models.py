"""
Shared data models for the chunking / storage pipeline.

These live in their own module so that `chunker.py` and `store.py` can both
depend on them without forming an import cycle.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    fingerprint: str
    chunk_index: int
    text: str
    start_page: int
    end_page: int
    source: str
    page_char_start: int
    page_char_end: int
    overlap_source: Optional[str] = None


@dataclass
class ChunkConfig:
    chunk_tokens: int = 512
    overlap_pct: float = 0.15
    min_chunk_tokens: int = 64
