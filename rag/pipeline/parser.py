import logging
import re
import time
from pathlib import Path

import opendataloader_pdf

from common.paths import OUTPUT_DIR, parsed_markdown_path
from fingerprint.hash import hash_file

logger = logging.getLogger(__name__)


def parse(file_path: str) -> str:
    """
    Parse a PDF, TXT, or MD file into page-marked markdown at
    `output/<fingerprint>.md` and return the document's SHA-256 fingerprint.
    Re-parsing the same file is a cache hit (the fingerprint is returned
    without re-conversion).
    """
    logger.info("parse start file=%s", file_path)
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(path)
    elif suffix in (".txt", ".md"):
        return _parse_text(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _parse_pdf(path: Path) -> str:
    start_time = time.perf_counter()
    fingerprint = hash_file(path)
    hash_elapsed = time.perf_counter() - start_time
    logger.info(
        "parse pdf hashed file=%s fingerprint=%s elapsed=%.2fs",
        path,
        fingerprint,
        hash_elapsed,
    )

    output_file = parsed_markdown_path(fingerprint)
    if output_file.exists():
        logger.info(
            "parse pdf cache hit fingerprint=%s output=%s",
            fingerprint,
            output_file,
        )
        return fingerprint

    logger.info("parse pdf converting file=%s output_dir=%s", path, OUTPUT_DIR)
    convert_start = time.perf_counter()
    opendataloader_pdf.convert(
        input_path=[str(path)],
        markdown_page_separator="\n\n<!-- page: %page-number% -->\n\n",
        output_dir=f"{OUTPUT_DIR}/",
        format="markdown"
    )
    default_output = OUTPUT_DIR / f"{path.stem}.md"
    default_output.rename(output_file)
    sanitize_markdown(output_file)
    logger.info(
        "parse pdf done fingerprint=%s output=%s elapsed=%.2fs",
        fingerprint,
        output_file,
        time.perf_counter() - convert_start,
    )
    return fingerprint


def _parse_text(path: Path) -> str:
    start_time = time.perf_counter()
    fingerprint = hash_file(path)
    hash_elapsed = time.perf_counter() - start_time
    logger.info(
        "parse text hashed file=%s fingerprint=%s elapsed=%.2fs",
        path,
        fingerprint,
        hash_elapsed,
    )

    output_file = parsed_markdown_path(fingerprint)
    if output_file.exists():
        logger.info(
            "parse text cache hit fingerprint=%s output=%s",
            fingerprint,
            output_file,
        )
        return fingerprint

    # opendataloader-pdf only accepts PDFs (the JAR rejects anything else),
    # so .txt/.md is wrapped in a single page marker here and written directly
    # to the same output/<fp>.md contract the chunker, store, and report
    # writer already expect. One page marker keeps page numbers / offsets /
    # raw re-slicing behaving identically across PDF and text inputs.
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_start = time.perf_counter()
    raw = path.read_text(encoding="utf-8")
    output_file.write_text(f"<!-- page: 1 -->\n\n{raw}", encoding="utf-8")
    sanitize_markdown(output_file)
    logger.info(
        "parse text done fingerprint=%s output=%s elapsed=%.2fs",
        fingerprint,
        output_file,
        time.perf_counter() - write_start,
    )
    return fingerprint


def sanitize_markdown(path: Path) -> None:
    text = path.read_text(encoding="utf-8").strip()
    before = len(text)
    text = re.sub(r'\n{2,}', '\n', text)
    after = len(text)
    path.write_text(text, encoding="utf-8")
    logger.info(
        "sanitize_markdown path=%s chars_before=%d chars_after=%d",
        path,
        before,
        after,
    )
