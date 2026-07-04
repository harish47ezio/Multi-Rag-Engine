import logging
import time
from pathlib import Path
from typing import List

import numpy as np

from common.paths import (
    INDEX_ANNOY_FILENAME,
    INDEX_HNSW_FILENAME,
    INDEX_IVF_FILENAME,
    INDEX_LSH_FILENAME,
    VECTORS_FILENAME,
    doc_storage_dir,
)
from rag.factory.instance import EmbeddingInstance
from rag.pipeline.models import Chunk
from rag.search.ann.ann import ANNSearcher
from rag.search.ann.annoy import AnnoySearcher
from rag.search.ann.ivf import IVFSearcher
from rag.search.ann.lsh import LSHSearcher

logger = logging.getLogger(__name__)


def index(chunks: List[Chunk], instance: EmbeddingInstance) -> None:
    """
    1. Embed all chunks via `instance.embed`
    2. Build each searcher index using `instance.metric` and save to disk

    Storage layout: every on-disk artifact for one (document, model) lives
    under `storage/{fingerprint}/{model_key}/`. The model_key partition
    matters because chunk boundaries differ per tokenizer (hence per
    model), so vectors and indexes built under model A are not compatible
    with model B even for the same document.

    Searchers themselves know nothing about fingerprint or model_key —
    the caller scopes them by directory.
    """
    if not chunks:
        raise ValueError("index() requires non-empty chunks.")

    fingerprints = {c.fingerprint for c in chunks}
    if len(fingerprints) != 1:
        raise ValueError(
            f"index() expects chunks from one document; got {len(fingerprints)}."
        )
    fp = next(iter(fingerprints))
    model_key = instance.model_key

    doc_dir = doc_storage_dir(fp, model_key)
    doc_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "index start chunks=%d fingerprint=%s model_key=%s dir=%s",
        len(chunks),
        fp,
        model_key,
        doc_dir,
    )

    vectors_path = doc_dir / VECTORS_FILENAME
    chunk_indices: List[int] = [c.chunk_index for c in chunks]

    # `vectors_rebuilt` forces every downstream index to be rebuilt even if a
    # stale index file already exists — otherwise indexes could point at
    # embeddings that no longer match the current chunk set.
    vectors_rebuilt = False
    vectors: List[List[float]] = []

    if vectors_path.exists():
        vectors_np = np.load(vectors_path)
        if len(vectors_np) == len(chunks):
            vectors = vectors_np.tolist()
            logger.info(
                "index vectors loaded from cache path=%s count=%d dim=%d",
                vectors_path,
                len(vectors),
                len(vectors[0]) if vectors else 0,
            )
        else:
            logger.warning(
                "index vectors cache stale path=%s cached=%d chunks=%d — re-embedding",
                vectors_path,
                len(vectors_np),
                len(chunks),
            )

    if not vectors:
        texts = [c.text for c in chunks]
        logger.info("index embedding chunks count=%d", len(texts))
        embed_start = time.perf_counter()
        vectors_np = np.array(instance.embed(texts), dtype=np.float32)

        np.save(vectors_path, vectors_np)
        vectors = vectors_np.tolist()
        vectors_rebuilt = True
        logger.info(
            "index embedding done count=%d dim=%d elapsed=%.2fs",
            len(vectors),
            len(vectors[0]) if vectors else 0,
            time.perf_counter() - embed_start,
        )

    metric = instance.metric
    dim = len(vectors[0])

    # ANN and IVF learn dim from the input vectors at index() time, so their
    # constructors don't take dim. Annoy and LSH need dim up front.
    # When vectors were just rebuilt, force every index to rebuild too so they
    # can never reference stale embeddings.
    hnsw_path = doc_dir / INDEX_HNSW_FILENAME
    if vectors_rebuilt or not hnsw_path.exists():
        _build_and_save(
            ANNSearcher(metric=metric),
            vectors,
            chunk_indices,
            hnsw_path,
            label="hnsw",
        )
    else:
        logger.info("index hnsw skipped (already exists) path=%s", hnsw_path)

    annoy_path = doc_dir / INDEX_ANNOY_FILENAME
    if vectors_rebuilt or not annoy_path.exists():
        _build_and_save(
            AnnoySearcher(metric=metric, dim=dim),
            vectors,
            chunk_indices,
            annoy_path,
            label="annoy",
        )
    else:
        logger.info("index annoy skipped (already exists) path=%s", annoy_path)

    ivf_path = doc_dir / INDEX_IVF_FILENAME
    if vectors_rebuilt or not ivf_path.exists():
        _build_and_save(
            IVFSearcher(metric=metric),
            vectors,
            chunk_indices,
            ivf_path,
            label="ivf",
        )
    else:
        logger.info("index ivf skipped (already exists) path=%s", ivf_path)

    lsh_path = doc_dir / INDEX_LSH_FILENAME
    if vectors_rebuilt or not lsh_path.exists():
        _build_and_save(
            LSHSearcher(metric=metric, dim=dim),
            vectors,
            chunk_indices,
            lsh_path,
            label="lsh",
        )
    else:
        logger.info("index lsh skipped (already exists) path=%s", lsh_path)

    logger.info(
        "index done chunks=%d fingerprint=%s model_key=%s dir=%s",
        len(chunks),
        fp,
        model_key,
        doc_dir,
    )

def _build_and_save(
    searcher,
    vectors: List[List[float]],
    chunk_indices: List[int],
    path: Path,
    label: str = "",
) -> None:
    logger.info("index %s building count=%d", label, len(chunk_indices))
    start = time.perf_counter()
    searcher.index(vectors, chunk_indices)
    searcher.save(str(path))
    logger.info(
        "index %s saved path=%s elapsed=%.2fs",
        label,
        path,
        time.perf_counter() - start,
    )
