import time

import numpy as np

from common.paths import (
    INDEX_ANNOY_FILENAME,
    INDEX_HNSW_FILENAME,
    INDEX_IVF_FILENAME,
    INDEX_LSH_FILENAME,
    VECTORS_FILENAME,
    doc_storage_dir,
)
from rag.factory.instance import Instance
from rag.search.ann.ann import ANNSearcher
from rag.search.ann.annoy import AnnoySearcher
from rag.search.ann.ivf import IVFSearcher
from rag.search.ann.lsh import LSHSearcher
from rag.search.knn.knn_searcher import KNNSearcher


def embed_query(instance: Instance, query: str):
    """Embed a query through the chosen Instance and return a single vector."""
    return instance.embed([query])[0]


def run_search(
    name: str,
    query_vector,
    query: str,
    fingerprint: str,
    instance: Instance,
    k: int = 5

):
    """
    Load the searcher for (`fingerprint`, `instance.model_key`) and run a top-5 search.

    All on-disk artifacts live under `storage/{fingerprint}/{model_key}/`
    (see indexer). Searchers themselves don't know about either — this
    caller scopes them by directory and attaches the fingerprint /
    model_key back to results when logging. The distance metric for
    every searcher comes from the Instance, never hard-coded here.
    """
    dim = len(query_vector)
    doc_dir = doc_storage_dir(fingerprint, instance.model_key)
    metric = instance.metric

    if name == "ANNSearcher":
        searcher = ANNSearcher(metric)
        searcher.load(str(doc_dir / INDEX_HNSW_FILENAME))
    elif name == "IVFSearcher":
        searcher = IVFSearcher(metric)
        searcher.load(str(doc_dir / INDEX_IVF_FILENAME))
    elif name == "LSHSearcher":
        searcher = LSHSearcher(metric, dim)
        searcher.load(str(doc_dir / INDEX_LSH_FILENAME))
    elif name == "AnnoySearcher":
        searcher = AnnoySearcher(metric, dim)
        searcher.load(str(doc_dir / INDEX_ANNOY_FILENAME))
    elif name == "KNNSearcher":
        # KNN is not persisted by the indexer — rebuild from cached vectors.
        # chunk_index aligns with vectors-array position because the indexer
        # embeds chunks in chunk_index order.
        searcher = KNNSearcher(metric)
        vectors = np.load(doc_dir / VECTORS_FILENAME).tolist()
        chunk_indices = list(range(len(vectors)))
        searcher.index(vectors, chunk_indices)
    else:
        raise ValueError(f"Unknown searcher '{name}'")

    search_start = time.perf_counter()
    results = searcher.search(query_vector, query, k)
    search_elapsed = time.perf_counter() - search_start

    return search_elapsed, results
