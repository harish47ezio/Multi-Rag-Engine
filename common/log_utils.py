"""
Shared logging helpers for the Multi RAG Engine.

Two responsibilities:

1. `setup_logging(level)` — one-shot configuration call for any entry-point
   script (e.g. alpha_test.py). Idempotent so it can be called from multiple
   alpha-test files in the same process without duplicating handlers.

2. `preview(text, n)` — produce a single-line, length-capped view of a string
   so log lines stay readable. Use this anywhere you would otherwise log a
   chunk body, a tokenized array, a prompt, or any user content.

Project convention: every module gets `logger = logging.getLogger(__name__)`
at the top, never imports a global logger. Levels are controlled centrally.
"""

import logging
from typing import Iterable

_LOG_FORMAT = "%(asctime)s %(levelname)-5s %(name)-32s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_configured: bool = False


def setup_logging(level: str = "INFO") -> None:
    """
    Configure root logging once for the process.

    Safe to call from multiple entry-points; subsequent calls only adjust
    the level on the existing root handler and do not stack new handlers.
    """
    global _configured

    root = logging.getLogger()
    resolved_level = getattr(logging, level.upper(), logging.INFO)

    if _configured:
        root.setLevel(resolved_level)
        for handler in root.handlers:
            handler.setLevel(resolved_level)
        return

    logging.basicConfig(
        level=resolved_level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
    )

    # Tame chatty third-party loggers so our INFO line-rate stays signal.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    _configured = True


def preview(text: str, n: int = 60) -> str:
    """
    Return a single-line preview of `text` capped at `n` characters.

    Collapses internal whitespace so the line stays compact even when the
    source contains newlines or runs of spaces. Long values are shown as
    "<first n-20 chars>…<last 20 chars>" so both ends carry signal.
    """
    if text is None:
        return "<None>"
    flat = " ".join(str(text).split())
    if len(flat) <= n:
        return flat
    head = flat[: max(1, n - 20)]
    tail = flat[-20:]
    return f"{head}…{tail}"


def count_chars(items: Iterable[str]) -> int:
    """Total character count across an iterable of strings (for log lines)."""
    return sum(len(s) for s in items)
