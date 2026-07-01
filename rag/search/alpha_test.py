import logging
import time

import numpy as np
import truststore

truststore.inject_into_ssl()

from common.log_utils import setup_logging
from rag.factory.instance import Instance
from rag.search.ann.ann import ANNSearcher
from rag.search.ann.annoy import AnnoySearcher
from rag.search.ann.ivf import IVFSearcher
from rag.search.ann.lsh import LSHSearcher
from rag.search.knn.knn_searcher import KnnSearcher

setup_logging("INFO")
logger = logging.getLogger(__name__)


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
    doc_dir = f"storage/{fingerprint}/{instance.model_key}"
    metric = instance.metric

    if name == "ANNSearcher":
        searcher = ANNSearcher(metric)
        searcher.load(f"{doc_dir}/index_hnsw.bin")
    elif name == "IVFSearcher":
        searcher = IVFSearcher(metric)
        searcher.load(f"{doc_dir}/index_ivf.faiss")
    elif name == "LSHSearcher":
        searcher = LSHSearcher(metric, dim)
        searcher.load(f"{doc_dir}/index_lsh.pkl")
    elif name == "AnnoySearcher":
        searcher = AnnoySearcher(metric, dim)
        searcher.load(f"{doc_dir}/index_annoy.ann")
    elif name == "KnnSearcher":
        # KNN is not persisted by the indexer — rebuild from cached vectors.
        # chunk_index aligns with vectors-array position because the indexer
        # embeds chunks in chunk_index order.
        searcher = KnnSearcher(metric)
        vectors = np.load(f"{doc_dir}/vectors.npy").tolist()
        chunk_indices = list(range(len(vectors)))
        searcher.index(vectors, chunk_indices)
    else:
        raise ValueError(f"Unknown searcher '{name}'")

    search_start = time.perf_counter()
    results = searcher.search(query_vector, query, k)
    search_elapsed = time.perf_counter() - search_start

    # logger.info("run_search name=%s elapsed=%.4fs hits=%d", name, search_elapsed, len(results))
    # for rank, (chunk_index, score) in enumerate(results, start=1):
    #     logger.info(
    #         "run_search name=%s rank=%d score=%.4f fp=%s model=%s chunk_index=%d",
    #         name,
    #         rank,
    #         score,
    #         fingerprint,
    #         instance.model_key,
    #         chunk_index,
    #     )
    return search_elapsed, results
