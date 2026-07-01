import logging
import time
from pathlib import Path
from typing import List

import numpy as np

from rag.factory.instance import Instance
from rag.pipeline.models import Chunk
from rag.search.ann.ann import ANNSearcher
from rag.search.ann.annoy import AnnoySearcher
from rag.search.ann.ivf import IVFSearcher
from rag.search.ann.lsh import LSHSearcher
from rag.search.base_search import BaseSearcher

logger = logging.getLogger(__name__)

STORAGE = Path("storage")


def index(chunks: List[Chunk], instance: Instance) -> None:
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

    doc_dir = STORAGE / fp / model_key
    doc_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "index start chunks=%d fingerprint=%s model_key=%s dir=%s",
        len(chunks),
        fp,
        model_key,
        doc_dir,
    )

    vectors_path = doc_dir / "vectors.npy"
    chunk_indices: List[int] = [c.chunk_index for c in chunks]

    if not vectors_path.exists():
        texts = [c.text for c in chunks]
        logger.info("index embedding chunks count=%d", len(texts))
        embed_start = time.perf_counter()
        vectors = np.array(instance.embed(texts), dtype=np.float32)

        np.save(vectors_path, vectors)
        vectors = vectors.tolist()
        logger.info(
            "index embedding done count=%d dim=%d elapsed=%.2fs",
            len(vectors),
            len(vectors[0]) if vectors else 0,
            time.perf_counter() - embed_start,
        )

    else:
        vectors_np = np.load(vectors_path)
        vectors: List[List[float]] = vectors_np.tolist()
        logger.info(
            "index vectors loaded from cache path=%s count=%d dim=%d",
            vectors_path,
            len(vectors),
            len(vectors[0]) if vectors else 0,
        )

    metric = instance.metric
    dim = len(vectors[0])

    # ANN and IVF learn dim from the input vectors at index() time, so their
    # constructors don't take dim. Annoy and LSH need dim up front.
    hnsw_path = doc_dir / "index_hnsw.bin"
    if not hnsw_path.exists():
        _build_and_save(
            ANNSearcher(metric=metric),
            vectors,
            chunk_indices,
            hnsw_path,
            label="hnsw",
        )
    else:
        logger.info("index hnsw skipped (already exists) path=%s", hnsw_path)

    annoy_path = doc_dir / "index_annoy.ann"
    if not annoy_path.exists():
        _build_and_save(
            AnnoySearcher(metric=metric, dim=dim),
            vectors,
            chunk_indices,
            annoy_path,
            label="annoy",
        )
    else:
        logger.info("index annoy skipped (already exists) path=%s", annoy_path)

    ivf_path = doc_dir / "index_ivf.faiss"
    if not ivf_path.exists():
        _build_and_save(
            IVFSearcher(metric=metric),
            vectors,
            chunk_indices,
            ivf_path,
            label="ivf",
        )
    else:
        logger.info("index ivf skipped (already exists) path=%s", ivf_path)

    lsh_path = doc_dir / "index_lsh.pkl"
    if not lsh_path.exists():
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
